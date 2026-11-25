# CODE.md

Este CODE también está disponible en:\
[en-GB](CODE.en-GB.md), [en-US](CODE.md), [es-AR](CODE.es-AR.md), es-ES

Referencia técnica de llama.cpp-xdna, escrita para consultarse de forma quirúrgica, saltando directamente a la sección que se necesite. Cubre lo que este fork añade a llama.cpp, sobre todo el backend XDNA para la NPU de los procesadores AMD Ryzen AI; el resto del código es el de [llama.cpp](https://github.com/ggml-org/llama.cpp). Las mediciones en que se apoya cada decisión están en [XDNA2.md](XDNA2.md) (en inglés).

## Índice

1. [Arquitectura y decisiones de diseño](#1-arquitectura-y-decisiones-de-diseño)
2. [Mapa de módulos](#2-mapa-de-módulos)
3. [Índice de símbolos clave](#3-índice-de-símbolos-clave)
4. [Flujos principales](#4-flujos-principales)
5. [Mapa de entradas](#5-mapa-de-entradas)
6. [Análisis de impacto](#6-análisis-de-impacto)
7. [Puntos de extensión](#7-puntos-de-extensión)

## 1. Arquitectura y decisiones de diseño

### El fork

Un fork de llama.cpp con tres backends: XDNA (escrito para este proyecto; hoy para las NPU XDNA2), CPU y RPC; los demás backends de llama.cpp se eliminaron. Tiene seis arquitecturas de modelo: `qwen35` (Qwen3.5 0.8B, 2B, 4B, 9B y Qwen3.8 27B, un híbrido de atención lineal y atención, con una capa de predicción de varios tokens), `qwen35moe` (lo mismo con un bloque de expertos, Qwen3.6 35B-A3B), `qwen4exp` (Qwen3.8 Flash Next `512x56B`: Hyper-Connections de 4 flujos, atención dispersa QSA con `lightning_indexer` + `top_k_mask`, incrustaciones por capa PLE bajo demanda y 512 expertos de anchura 640), `granite` (Granite 4.2 denso), `flint` (Flint 40B-A2.5B, un transformer con ventana deslizante intercalada, sumideros de atención y un bloque de expertos) y `qwen3tts` (síntesis de voz Qwen3-TTS 12Hz `Base`, `CustomVoice` y `VoiceDesign`, junto con `libmtmd`). Cada cambio en un archivo que viene de llama.cpp lleva un comentario `llama.cpp-xdna`, para poder conservarlo al portar código de upstream.

### Un backend propio, sin XRT

El backend habla directamente con el controlador `amdxdna` de Linux, con sus ioctl (`ggml/src/ggml-xdna/uapi/amdxdna_accel.h`, copiado de Linux 7.3.4): no hace falta XRT, y Debian 13 no tiene ningún XRT que conozca la NPU. `XdnaDevice` es esa capa: abre `/dev/accel/accel0`, crea búferes que comparten el anfitrión y la NPU, crea y configura contextos, envía órdenes y las espera en el objeto de sincronización de DRM del contexto (en esta NPU no existe `WAIT_CMD`).

### La NPU y sus overlays

La NPU (AIE2P) es una matriz de 8 columnas: una tesela de interfaz (la *shim*, que da a la memoria), una tesela de memoria (512 KB) y 4 núcleos (64 KB de datos y 16 KB de programa cada uno) por columna, 32 núcleos en total. Lo que hace la matriz es un *overlay*: un PDI con la configuración de cada tesela y los programas de los núcleos, construido a partir de un diseño en IRON (Python, `ggml/src/ggml-xdna/overlays/designs/`) y de los kernels de los núcleos (C++ para AIE, `ggml/src/ggml-xdna/kernels/`) con mlir-aie y su compilador Peano.

Cambiar el overlay con el que corre una orden cuesta entre 2,0 y 2,5 ms, así que hay tres, cada uno en su propio `XdnaContext` compartiendo dispositivo, heap y búferes `SHARE` (`vContext`, `vGemvContext` y `vMoeContext` de `XdnaRunner`):

- **El overlay de los prompts** (`xdna2-gemm-bfp16.*`): multiplicación de matrices C = A·B con las bandas de A guardadas en las teselas de memoria (`bfpmm_resident.py`), el kernel en pasos de tres juegos de sumas (`kernels/xdna2-mm-bfp16-s3.cc`) decodificando `B` desde `bfp16`, `Q8_0` o `Q6_K` de 4 teselas en 3 objetos (`comp 1`), C escrito en bfloat16 por franjas de 64 columnas y la recurrencia de una gated delta net de un lote en los mismos búferes (`kernels/xdna2-gdn-prompt.cc`, admitiendo cualquier razón entera de cabezas `1 ≤ cabezas_V / cabezas_K ≤ 4`).
- **El overlay encadenado para generar tokens** (`xdna2-gemv-chain.*`, versión 9): de 1 a 8 filas por una matriz, en fases (`gemv_chain.py`, `kernels/xdna2-gemv-chain.cc`), con pesos descomprimidos en los núcleos desde `bfp16`, `Q4_K`, `Q5_K`, `Q6_K` o `Q8_0` (`cQ8_0`) y `cChain = 1.728` flotantes (`vGateMax = 1.088`). Una orden ejecuta hasta 4 fases, y cada fase toma su X de la Y de la fase anterior dentro de la matriz, con su prólogo (la norma) y su epílogo (la suma residual, la SwiGLU): una capa entera va en una orden.
- **El overlay de mezcla de expertos** (`xdna2-moe.*`, versión 1): hasta 1.136 trozos de 1 a 24 filas multiplicados por sus matrices de expertos elegidas en una sola orden (`moe_experts.py`, `kernels/xdna2-moe.cc`), con pesos descomprimidos en los núcleos desde `bfp16`, `Q4_K`, `Q5_K`, `Q5_1` (`cQ5_1`, incluido `Q5_0` convertido sin pérdida al cargar), `Q6_K` o `Q8_0` (`cQ8_0`), grupos de cola exactos en N (`cParamTailGroups`) y subbloques simétricos de 32 en K (`cParamTailSubs`), planes densos de 1 experto en caché (`aMoeDensePlans`) y SwiGLU opcional de dos tramos dentro del núcleo 0.

El anfitrión escribe en C++ el flujo de instrucciones de cada orden (las transferencias de las teselas de interfaz, los parámetros de los núcleos, las esperas) a partir de un manifiesto que escribe la compilación del overlay (`xdna-overlay.*`, `xdna-chain.*`, `xdna-moe.*`). Los flujos son iguales, operación a operación, que los de mlir-aie, y se comprobó; mientras corre un modelo no se ejecuta nada de Python.

### Formatos numéricos

- **Coma flotante por bloques** (bfp16ebs8): 8 valores de 8 bits que comparten un exponente, 9 bytes. Los operandos del overlay de los prompts van en bfp16, las sumas en flotantes de 32 bits, y C se redondea a bfloat16. En precisión, esto cuesta más o menos lo que cuesta Q8_0 en el procesador.
- **Los pesos del archivo**: para generar tokens y para los expertos enrutados, los núcleos descomprimen `Q4_K`, `Q5_K`, `Q6_K` y `Q8_0` tal como los guarda el archivo; los demás tipos van en bfp16.
- Los pesos se convierten una vez, la primera vez que se usan, en búferes del controlador (con empaquetado por franjas residente en L2 `fXdnaGemmPackBRows` y descuantizadores AVX2+F16C `fDequantizeQ8_0Avx2` / `fDequantizeQ6KAvx2`); el archivo sigue proyectado en memoria (`mmap`), los búferes de pesos del backend se crean sobre esa proyección (`fDeviceBufferFromHostPtr`), y `MADV_DONTNEED` libera las páginas 2D densas (`fNoteDenseCached`) y 3D de expertos (`fReleaseMappedExpertPages`) en cuanto están listas sus copias para la NPU.

### Qué toma el backend de un grafo

El dispositivo es un acelerador (`GGML_BACKEND_DEVICE_TYPE_ACCEL`): llama.cpp lo pone delante de la CPU en cada contexto. Su tipo de búfer es memoria del anfitrión (`is_host`), para que llama.cpp le dé los pesos de las multiplicaciones que toma y la CPU pueda seguir leyéndolos; `fDeviceSupportsOp` dice qué nodos toma, y `fGraphCompute` ejecuta una parte de un grafo y reconoce patrones de nodos para hacerlos de una vez:

- **Un lote de un prompt** (40 tokens o más): las multiplicaciones que leen las mismas filas, en una orden (`fMulMat`); una norma RMS y sus pesos, aplicados al convertir las filas; la SwiGLU, calculada al leer la puerta y la subida, partida en mitades desde `vHalf >= 1.024` con envío temprano del trozo 0 de `ffn_down` (`vEarlySent`) y escrita directamente como las filas de `ffn_down` (`fMulMatGlu`); la suma residual, leída con C; el `CONCAT` de la convolución corta, escrito por la multiplicación, y la convolución corta con su SiLU más el preempaquetado directo de `k`, `q` y `v` en `bf16` con sus normas L2 al leer la C de `attn_qkv` (`fConvFold`, `fConvStrips`, `mPrepareGdnB`); la norma con puerta de la atención lineal, convertida directamente desde la salida de la recurrencia en las teselas de `A` de `ssm_out` (`fGatedNorm`, `fXdnaGdnUnpackGatedToA`); la puerta sigmoide de atención completa fusionada en el anfitrión (`fAttnGateStep`); la recurrencia de la atención lineal en la NPU (`fGatedDeltaNet`, razones `1:1` a `4:1`, con `0` cortes recurrentes gracias a `fStateOpTaken`); la atención como dos multiplicaciones por cada cabeza de claves, con la softmax en el anfitrión entre ellas (`fFlashAttention`).
- **La memoria de las copias**: cuando `MADV_DONTNEED` libera las páginas convertidas del archivo (sin `GGML_XDNA_WEIGHTS_GB` ni `GGML_XDNA_COPY_WEIGHTS`), los modelos densos colocan ambas copias hasta el 48 % de la memoria del anfitrión (`vDenseChainLimit`), o encadenan el 100 % de las capas de generación manteniendo estáticas las capas de prompt que caben y transmitiendo al vuelo solo el resto en dos búferes reutilizables hasta el 72 % (`vDenseTotalLimit`, `vStreamPrompt`, como en Qwen3.8 27B; `GGML_XDNA_PROMPT_FORMAT=q6k` / `q8_0` guarda en su lugar una única copia estática comprimida), y las mezclas de expertos usan hasta el 78 % (`vMoeWeightsLimit`). Por encima de esos límites o si se fija `GGML_XDNA_WEIGHTS_GB`, las capas restantes se quedan en el procesador (`XdnaPlacement::vPartial`).
- **Una mezcla de expertos** (`qwen35moe`, `flint`): las capas enteras de expertos enrutados (`MUL_MAT_ID`) van a `xdna2-moe` (`fPlaceMoeWeights`, `fMoeTaken`, `fRunMoeStep`), fusionando `{up_exps, gate_exps} + SwiGLU`, absorbiendo el reescalado `build_amax_src1_f16` de 10 nodos de Flint (`fMatchMoeAmax`) para continuar en `down_exps` desde la Y conservada (`mRunMoeFromKept`), plegando el experto compartido y su puerta escalar sigmoide como 1 o 2 expertos virtuales en cualquier tamaño de lote, incluidos los de 9 a 39 filas (`fMulMatTaken` `vShexpOnMoe`, `fMatchMoeSharedExpert`, `fEnsureMoeWithShared`, que también reutiliza las matrices `dMoeMatrices` cargadas con experto compartido cuando se consultan sin él) y fusionando la reducción ponderada más la `ADD` residual en la lectura de Y (`fMatchMoeTail`, `fXdnaMoeReadY`), además de liberar las páginas del archivo proyectado de los expertos 3D (`fReleaseMappedExpertPages`) cuando `GGML_XDNA_MOE_MIN_TOKENS` vale 1. Su enrutador discreto se queda por defecto en el procesador (`fIsRouter`), o pasa a XDNA con `GGML_XDNA_FULL`; al generar tokens (de 1 a 8 filas), las proyecciones densas de atención (`fEnsureMoeDenseGroup`, `fXdnaMoeReadYTargets`), los bloques feed-forward densos iniciales (`fRunDenseMoeFfn`) y `output.weight` corren también en `xdna2-moe` con 0 cambios de overlay (`GGML_XDNA_NO_MOE_DENSE=1` deja la generación densa y del experto compartido en el procesador; cuando los expertos de 1 token corren en el procesador con `GGML_XDNA_MOE_MIN_TOKENS` ≥ 2 o `GGML_XDNA_NO_MOE=1`, `fSharedExpertChain` encadena el experto compartido, su puerta y las sumas residuales con la norma y las proyecciones de la capa siguiente en `xdna2-gemv-chain`, también con 0 cambios).
- **Generar un token** (de 1 a 8 filas): cada bloque feed-forward, con la norma y las proyecciones de la capa siguiente, en una orden del overlay encadenado (`fFeedForward`); la última capa, con la norma de salida y los logits. Con `GGML_XDNA_FULL=1` (o `2`), `fFullOpTaken` y `fFullOpOnHost` reclaman todas las operaciones restantes para que prompts y generación se ejecuten en un solo subgrafo XDNA (`splits = 1`).

### La memoria del anfitrión y la de la NPU

La NPU no es coherente con la caché del procesador. Lo que escribió el anfitrión sale de la caché antes de que lo lea una orden; lo que escribió la NPU se invalida en la caché antes de que lo lea el anfitrión; un búfer nuevo sale de la caché antes de que lo escriba la NPU (el kernel pone a cero sus páginas a través de la caché, y esos ceros volverían encima de los resultados); y en un búfer que escriben los dos, el anfitrión vacía hasta el final de la página de lo que escribió y una página más (la precarga deja modificada la línea siguiente). `fXdnaFlushCache` vacía desde el proceso (`clflushopt`), y `XdnaDevice::mSyncBuffer`, a través del controlador.

Una orden que no termina deja la NPU inutilizable hasta que se apaga el ordenador; el runner no vuelve a enviar ninguna orden después de una así (`vBroken`), y el ancho de un contexto tiene que ser el de la partición para la que se compiló el overlay.

### Cambios fuera del backend

- `src/llama-context.cpp`, `llama_context::graph_get_cb` y `resolve_fused_ops`: el planificador solo pasa una operación a un acelerador desde sus pesos o desde los nodos vecinos; esta devolución de llamada le da al backend XDNA, en una capa del procesador, la recurrencia de un lote, las normas L2 de q y k, la atención, la puerta sigmoide de atención completa, las normas (`norm` y `l_last`) del flujo residual y las vistas prefijo `VIEW` de desplazamiento cero sobre tensores de pesos (como la cabeza parcial `head_part` de `n_vocab_mtp`), cuando las toma, mientras que `resolve_fused_ops` acepta las operaciones fusionadas reclamadas por un acelerador `ACCEL` en capas de CPU (`accel_on_cpu`).
- `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp` y `src/models/granite.cpp`: las proyecciones de una capa seguidas en el grafo (y los nodos unitarios `SCALE(1.0)` omitidos en `granite.cpp` cuando `f_logit_scale == 1.0f` o `f_residual_scale == 1.0f`), la entrada de la convolución corta por tokens, la convolución corta antes que el estado recurrente, y los estados de la convolución y de la recurrencia leídos y escritos en su celda de la caché.
- `src/llama-graph.{h,cpp}`, `ggml/include/ggml.h`, `ggml/src/ggml.c`: `ggml_gated_delta_net_set_state_in_place` y `llm_graph_rs_in_place`.
- `src/llama-memory-recurrent.cpp` y `src/llama-kv-cache.cpp`: los estados recurrentes y las cachés KV en `F16` (`<= 4 GiB`, incluida `llama_kv_cache_iswa`) en búferes compartidos `XDNA_State` del controlador (`ggml_backend_dev_get_state_buft`, `llama_kv_cache_accel_buft`), con seguimiento de celdas limpias (`XdnaShared::sCleanStates`).
- Las mezclas de expertos, que el fork había quitado, portadas de upstream para `qwen35moe` (`src/models/qwen35moe.cpp`, generado desde `src/models/qwen35.cpp` con `_/port-tools/make_qwen35moe.py`): las claves, los tensores y los parámetros de los expertos (`src/llama-arch.*`, `src/llama-hparams.*`, `src/llama-model.*`), la prueba de `MUL_MAT_ID` que coloca sus búferes (`src/llama-model-loader.cpp`), `build_lora_mm_id` y un `build_moe_ffn` reducido (`src/llama-graph.*`), leyendo por adelantado `conv_states` de la siguiente capa recurrente justo tras `attn_post_norm` para que ningún corte `GET_ROWS` del procesador quede entre `moe_out` y `ffn_gate_shexp`.
- `flint` (`src/models/flint.cpp`), portada de upstream con lo que necesita y el fork había quitado: la caché KV con ventana deslizante intercalada (`src/llama-kv-cache-iswa.{h,cpp}`, `llm_graph_input_attn_kv_iswa`, `build_attn_inp_kv_iswa` y su `build_attn` en `src/llama-graph.*`, la rama de `llama_model::create_memory`), los sumideros de atención, la RoPE solo en las capas que la tienen, la base global de la RoPE en sus capas con ventana (`llama_model::get_rope_freq_base`), el sesgo de la elección de los expertos y el reescalado de la activación de sus expertos antes de su proyección de bajada (`build_amax_src1_f16`).
- `qwen4exp` (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.{h,cpp}`), portada de upstream con validación de hiperparámetros mediante excepciones, expansión conjunta de proyecciones (`ple_key` + `ple_value`, `wq` + `wk` + `wv` + `index_q_proj` + `index_k_proj`), norma con puerta `SIGMOID` en atención lineal (`fXdnaSigmoidMul`), modo `kv_only` (`n_outputs == 0`) y cabeza parcial `n_vocab_mtp` en `graph_mtp` (con reserva de presupuesto `GGML_XDNA_MTP_FIRST_LAYER` / `GGML_XDNA_MTP_LAYERS` y `cWeightViewKeyTag` en `ggml-xdna.cpp`), desactivación automática de `CPU_REPACK` y del `POSIX_MADV_WILLNEED` completo en modelos `mmap` mayores que la RAM (`src/llama-model.cpp`) y gestión `MADV_WILLNEED` / `POSIX_MADV_RANDOM` de expertos activos en unidades USB con *readahead* sobredimensionado (`llama_fd_read_ahead_kb`, `ggml/src/ggml-cpu/ggml-cpu.c`).
- `qwen3tts` (`src/models/qwen3tts.cpp`, `tools/mtmd/`, `tools/tts/tts.cpp`, `common/arg.cpp`, `vendor/miniaudio/`, `_/port-tools/convert_qwen3tts_to_gguf.py`): el Talker de 28 capas (`LLM_ARCH_QWEN3TTS`, M-RoPE intercalada, `{Qcur, Kcur, Vcur}` y `output` expandidas nada más crearlas para `fFeedForward`), más una `libmtmd` reducida solo a Qwen3-TTS (`mtmd_select_device` elige `XDNA`, `mtmd_gen_audio_get_voice` descubre las voces `<|spk_*|>` de `CustomVoice`, `--tts-instruct` antepone `<|im_start|>user\n...\n` en `VoiceDesign` y `1.7B-CustomVoice`, `gen_code_head_2d[0..14]` divide `a.gen.code.head.weight` en tensores 2D independientes para encadenar `code_gen` en `xdna2-gemv-chain`, y `conv1d_w_2d` / `conv1d_tr_w_2d` dan a `code2wav` pesos 2D independientes de convolución en `xdna2-gemm-bfp16`).
- `ggml/src/ggml-cpu/ops.cpp`: `SSM_CONV`, `GATED_DELTA_NET` y `RMS_NORM` más rápidos en el procesador, con los mismos resultados bit a bit; `tools/llama-bench/llama-bench.cpp` y `examples/speculative-simple/speculative-simple.cpp` admiten `--no-xdna` (`GGML_XDNA_DISABLE=1` antes de inicializar los backends) para medir el procesador solo.
- Seguridad: el servidor RPC valida lo que le manda un cliente y puede pedir un token (HMAC-SHA256); `llama-server` limita CORS a localhost y comprueba la cabecera `Host`; el lector de GGUF y el cargador rechazan los archivos malformados. Ver [SECURITY.md](../SECURITY.md) (en inglés).

## 2. Mapa de módulos

| Módulo | Ruta | Responsabilidad | Depende de | Usado por |
|---|---|---|---|---|
| Capa de dispositivo | `ggml/src/ggml-xdna/xdna-device.{h,cpp}`, `uapi/` | ioctl de `amdxdna`: búferes, contextos, órdenes, esperas, vaciados de caché | el controlador del kernel | runner, `llama-xdna-*` |
| Flujos de instrucciones | `xdna-insts.{h,cpp}` | Operaciones de un flujo; transferencias de las teselas de interfaz (`XdnaShimDma`) | — | gemm, gemv, chain |
| Manifiestos | `xdna-overlay.{h,cpp}` | Lo que ofrece un overlay de los prompts (teselas, bandas, kernel, `gdn`, `rows`) | — | gemm, runner, backend |
| Multiplicación de los prompts | `xdna-gemm.{h,cpp}` | Planes y flujos; disposición de A, B y C; conversiones (filas, SwiGLU, norma con puerta, atención) | insts, overlay, bfp16, glu, workers | runner |
| Overlay de generación | `xdna-gemv.{h,cpp}` | El overlay sin encadenar: formas, flujos, formatos de los pesos (con empaquetado AVX2 para `Q6_K`) | insts | runner, chain, moe |
| Overlay encadenado | `xdna-chain.{h,cpp}` | Fases, órdenes, sus flujos y el búfer del anfitrión | insts, gemv | runner |
| Overlay de mezcla de expertos | `xdna-moe.{h,cpp}` | Formas, empaquetado de expertos, flujos en anillo de 3 ranuras (`fXdnaMoePlan`), codificación con norma por fila, transcodificación de Y a X y lectura ponderada o por destinos en paralelo | insts, gemv, bfp16, workers | runner, backend |
| Recurrencia de un prompt | `xdna-gdnp.{h,cpp}` | Entradas y salidas de la recurrencia de un lote | bfp16, workers | runner |
| Coma flotante por bloques | `xdna-bfp16.{h,cpp}` | Codificador (AVX2), decodificador, bfloat16 y flotantes de 16 bits | — | todo lo que convierte |
| Aritmética del anfitrión | `xdna-glu.{h,cpp}` | SwiGLU, escalas RMS, puntuaciones y exponenciales de la atención, como las calcula el backend de la CPU | bfp16 | gemm, backend |
| Hilos del anfitrión | `xdna-workers.{h,cpp}` | Bucles repartidos (OpenMP), con los demás hilos dormidos mientras trabaja la NPU | OpenMP | gemm, runner |
| Runner | `xdna-runner.{h,cpp}` | Abre la NPU con los tres overlays; convierte, envía y lee multiplicaciones, recurrencias, órdenes encadenadas, la atención y los expertos enrutados | todo lo anterior | backend, herramientas |
| Backend de ggml | `ggml-xdna.cpp`, `ggml/include/ggml-xdna.h` | Dispositivo, tipos de búfer, `supports_op`, cálculo del grafo y sus fusiones | runner, ggml | llama.cpp |
| Kernels | `ggml/src/ggml-xdna/kernels/*.cc` | Programas de los núcleos | Peano, AIE API | overlays |
| Overlays | `ggml/src/ggml-xdna/overlays/` | Los tres overlays compilados, sus diseños y `build-overlays.sh` | mlir-aie, Peano | CMake, backend |
| Herramientas | `tools/xdna-*/` | `llama-xdna-info`, `-run`, `-insts`, `-gemm`, `-gemv`, `-probe` | runner, device | desarrolladores |
| Síntesis de voz (`libmtmd`, `llama-tts`) | `tools/mtmd/`, `tools/tts/`, `vendor/miniaudio/`, `_/port-tools/convert_qwen3tts_to_gguf.py` | Codificador `ECAPA-TDNN` (`Base`), voces `<|spk_*|>` (`CustomVoice`), condicionamiento `--tts-instruct` (`VoiceDesign` / `1.7B-CustomVoice`), predictor `code_gen` de 16 pasos, vocoder DAC `code2wav` a 24 kHz y conversor a GGUF | llama, ggml, backend | `llama-tts` |
| Instaladores | `deploy/install-update-reinstall-{debian,alpine}.sh` | Compilan e instalan desde las fuentes | git, cmake, Node.js | usuarios |

## 3. Índice de símbolos clave

Rutas bajo `ggml/src/ggml-xdna/` salvo que se indique otra cosa.

| Símbolo | Archivo:línea | Qué hace |
|---|---|---|
| `fProbe` | `ggml-xdna.cpp:522` | Lee los ajustes (`GGML_XDNA_*`), los manifiestos y la NPU al registrarse el backend |
| `fShared` | `ggml-xdna.cpp:698` | Lo que comparten los backends de un proceso: el runner, las matrices convertidas, los búferes de estados |
| `fCloseIfIdle`, `fOpenRunner` | `ggml-xdna.cpp:776`, `:798` | Cierran la NPU cuando nada la necesita; la abren con los dos overlays |
| `fMulMatTaken` | `ggml-xdna.cpp:1514` | Si un `MUL_MAT` va a la NPU (pesos en un búfer del backend, filas, tipos) |
| `fNoteDenseCached` | `ggml-xdna.cpp:1693` | Libera las páginas 2D proyectadas con `MADV_DONTNEED` cuando existen ambas copias |
| `fPlaceWeights`, `fPlaceMoeWeights` | `ggml-xdna.cpp:1110`, `:6366` | Si un peso denso o una capa entera de expertos entra en el backend dentro del presupuesto (`vWeightsLimit` / `vMoeWeightsLimit`) |
| `fIsRouter`, `fSawExperts` | `ggml-xdna.cpp:1021`, `:1107` | El enrutador de una mezcla de expertos, que se queda en el procesador; si se ha visto alguno |
| `fMoeTaken`, `fRunMoeStep` | `ggml-xdna.cpp:6441`, `:7632` | Si un `MUL_MAT_ID` va a `xdna2-moe` y cómo corren `{gate, up} + SwiGLU + down`, los expertos compartidos y la reducción ponderada |
| `fMatchMoeSharedExpert`, `fEnsureMoeWithShared` | `ggml-xdna.cpp:7494`, `:7041` | Pliegan el experto compartido de una capa y su puerta escalar como expertos virtuales en las órdenes de `xdna2-moe` |
| `fEnsureMoeDensePair`, `fRunDenseMoeFfn` | `ggml-xdna.cpp:6985`, `:7991` | Bloques feed-forward densos de 1 a 8 filas en `xdna2-moe` sin cambiar de overlay |
| `fSharedExpertChain` | `ggml-xdna.cpp:5660` | Encadena el bloque de experto compartido y sus sumas residuales con la norma y las proyecciones de la capa siguiente en `xdna2-gemv-chain` |
| `fConvert`, `fMatrixOf` | `ggml-xdna.cpp:1860`, `:2086` | Los pesos de uno o más tensores convertidos para el overlay de los prompts, en trozos |
| `fGemvConvert` | `ggml-xdna.cpp:2181` | Lo mismo para el overlay de generación |
| `fMultiplyRows` | `ggml-xdna.cpp:2299` | De 1 a 8 filas en el overlay de generación |
| `fMultiply`, `fMulMat` | `ggml-xdna.cpp:2458`, `:2663` | Una multiplicación de muchas filas, con dos órdenes en vuelo; un grupo de multiplicaciones de las mismas filas |
| `fGdnTaken`, `fGatedDeltaNet` | `ggml-xdna.cpp:3045`, `:3299` | La recurrencia de un lote en la NPU |
| `fNormedQueries`, `fAttnTaken`, `fFlashAttention` | `ggml-xdna.cpp:3578`, `:3607`, `:3675` | La atención de un lote: si se parten Q y K, si se toma y cómo se ejecuta |
| `fConvTaken`, `fConvFold`, `fConvOnHost` | `ggml-xdna.cpp:4099`, `:4156`, `:4266` | La convolución corta de un lote: si se toma, si va con la lectura de C, y en el anfitrión |
| `fLinearTaken`, `fLinearAttention` | `ggml-xdna.cpp:3932`, `:4646` | La capa entera de atención lineal al generar (`GGML_XDNA_LINEAR`) |
| `fMulMatGlu` | `ggml-xdna.cpp:4904` | La puerta y la subida con su SwiGLU, en las filas de `ffn_down` |
| `fBackendFree` | `ggml-xdna.cpp:5119` | El informe de lo que hizo el backend |
| `fFeedForward` | `ggml-xdna.cpp:5298` | Una capa de un token en una orden del overlay encadenado |
| `fAttnGateStep` | `ggml-xdna.cpp:6025` | Fusiona en el anfitrión la puerta `SIGMOID + MUL` de atención completa |
| `fGatedNorm` | `ggml-xdna.cpp:6218` | La norma con puerta en las filas de `ssm_out` |
| `fFullOpTaken`, `fFullOpOnHost` | `ggml-xdna.cpp:8096`, `:8390` | Operaciones de pegamento del subgrafo único (`GGML_XDNA_FULL=1` / `2`, `splits = 1`) |
| `fGraphCompute` | `ggml-xdna.cpp:9312` | Ejecuta una parte de un grafo y reconoce sus patrones |
| `fBufferFree`, `fDeviceBufferFromHostPtr` | `ggml-xdna.cpp:10059`, `:10143` | Búferes de pesos; sobre la proyección del archivo |
| `fStateBufferTypeAlloc` | `ggml-xdna.cpp:10240` | Búferes del controlador para los estados recurrentes |
| `fDeviceSupportsOp` | `ggml-xdna.cpp:10344` | Las operaciones que toma el dispositivo |
| `ggml_backend_xdna_reg` | `ggml-xdna.cpp:10513` | El registro del backend |
| `XdnaRunner::mOpen`, `mSubmitGemvAndWait` | `xdna-runner.cpp:40`, `:273` | Abre `vContext`, `vGemvContext` y `vMoeContext`; envía y espera órdenes en `vGemvContext` |
| `XdnaRunner::mEnsure`, `mLoadWeightsRows` | `xdna-runner.cpp:304`, `:405` | Un búfer de al menos un tamaño, vaciado cuando es nuevo; empaquetado residente en L2 fila a fila |
| `XdnaRunner::mPackRows`, `mPrepare` | `xdna-runner.cpp:525`, `:567` | Filas en un búfer de A; el flujo y la orden de una multiplicación |
| `XdnaRunner::mSubmitPacked`, `mFinishPacked` | `xdna-runner.cpp:719`, `:745` | Una multiplicación enviada sin esperar, y su resultado |
| `XdnaRunner::mUnpackSwigluRows` | `xdna-runner.cpp:899` | La SwiGLU de dos C en las filas de la multiplicación siguiente |
| `XdnaRunner::mPackAttnP` | `xdna-runner.cpp:999` | La P de la atención, desde S, en las filas de P·V |
| `XdnaRunner::mLoadChainMatrix`, `mChainMultiplyRows` | `xdna-runner.cpp:1241`, `:1355` | Pesos para el overlay encadenado; filas por ellos |
| `XdnaRunner::mRunGdn` | `xdna-runner.cpp:1448` | La recurrencia de un lote, en hasta 4 órdenes en vuelo |
| `XdnaRunner::mRunChain` | `xdna-runner.cpp:1576` | Una orden de fases del overlay encadenado |
| `XdnaRunner::mLoadMoeMatrix`, `mRunMoe`, `mRunMoeFromKept` | `xdna-runner.cpp:1783`, `:1985`, `:2103` | Pesos de expertos y órdenes de `xdna2-moe`, incluida la continuación directa de Y a X |
| `XdnaDevice::mOpen`, `mCreateBuffer` | `xdna-device.cpp:171`, `:360` | La NPU; un búfer del controlador |
| `XdnaDevice::mConfigureContextUnits` | `xdna-device.cpp:505` | Varios overlays en un contexto |
| `XdnaDevice::mSubmit`, `mWait` | `xdna-device.cpp:558`, `:576` | Una orden; su espera |
| `fXdnaFlushCache` | `xdna-device.cpp:110` | Vaciado de la caché desde el proceso |
| `fXdnaGemmPlan`, `fXdnaGdnPlan` | `xdna-gemm.cpp:110`, `:395` | Flujos de una multiplicación y de la recurrencia de un lote |
| `fXdnaGemmShape`, `fXdnaGemmPackA`, `fXdnaGemmPackB`, `fXdnaGemmPackBRows` | `xdna-gemm.cpp:766`, `:798`, `:831`, `:880` | Tamaños con relleno; A y B en la disposición de la matriz (entera o franja a franja desde filas) |
| `fXdnaGemmPackSwiglu`, `fXdnaGemmPackAttnP` | `xdna-gemm.cpp:2002`, `:2299` | Una SwiGLU, o la P de la atención, convertida en las teselas de A |
| `fConvStrips` | `xdna-gemm.cpp:1453` | La convolución corta y su SiLU al leer C |
| `XdnaChainOverlay::mLoad`, `fXdnaChainPlan` | `xdna-chain.cpp:149`, `:583` | Manifiesto y flujo del overlay encadenado |
| `XdnaMoeOverlay::mLoad`, `fXdnaMoePlan`, `fXdnaMoeReadYTargets` | `xdna-moe.cpp:187`, `:476`, `:828` | Manifiesto, flujo en anillo de 3 ranuras y desempaquetado por destinos del overlay de expertos |
| `fXdnaGemvPackStream` | `xdna-gemv.cpp:706` | Pesos de un tramo, desde flotantes o desde los bloques de ggml |
| `XdnaOverlay::mLoad` | `xdna-overlay.cpp:36` | Manifiesto del overlay de los prompts |
| `XdnaShimDma::mPush` | `xdna-insts.cpp:136` | Una transferencia de un canal de la shim |
| `fXdnaBfp16EncodeNearest` | `xdna-bfp16.cpp:202` | El codificador rápido (AVX2) |
| `fXdnaGdnPack` | `xdna-gdnp.cpp:440` | Entradas de la recurrencia de un lote |
| `fXdnaSwiglu` | `xdna-glu.cpp:595` | SwiGLU con los vectores del backend de la CPU |
| `fXdnaConvToken`, `fXdnaConvRowsBf16` | `xdna-glu.cpp:692`, `:731` | Un token de la convolución corta, como lo calcula el backend de la CPU |
| `XdnaWorkers::mForEach` | `xdna-workers.cpp:32` | Un bucle en los hilos del anfitrión |
| `llama_context::resolve_fused_ops` | `src/llama-context.cpp:611` | Las sondas que deciden flash attention y la recurrencia fusionada |
| `graph_get_cb`, sus bloques `llama.cpp-xdna` | `src/llama-context.cpp:2850` | Operaciones que se dan al acelerador en las capas del procesador |
| `llama_memory_recurrent_accel_buft` | `src/llama-memory-recurrent.cpp:23` | Búferes de estados del acelerador |
| `llama_kv_cache_accel_buft` | `src/llama-kv-cache.cpp:125` | Búferes `XDNA_State` (`BO_SHARE`) para las cachés KV de las capas de CPU |
| `llama_model_flint` | `src/models/flint.cpp:10` | Flint: sus parámetros (comprobados), sus tensores y su grafo |
| `llama_model_qwen4exp`, `llama_memory_hybrid_idx` | `src/models/qwen4exp.cpp:26`, `src/llama-memory-hybrid-idx.h:18` | Qwen3.8 Flash Next (`qwen4exp`): Hyper-Connections, caché del indexador QSA, PLE bajo demanda y grafo de 512 expertos |
| `llama_model_qwen3tts` | `src/models/qwen3tts.cpp:7` | Qwen3-TTS Talker: sus parámetros (comprobados), sus tensores y su grafo encadenado |
| `mtmd_select_device`, `mtmd_graph_qwen3tts_gen::build` | `tools/mtmd/mtmd.cpp:154`, `tools/mtmd/models/qwen3tts-gen.cpp:654` | Elige `XDNA` para `libmtmd` (con voces `<|spk_*|>` y `--tts-instruct` en `mtmd.cpp`); construye `code_gen` (cabezas 2D) y `code2wav` (pesos 2D de convolución causal y traspuesta) |
| `llama_kv_cache_iswa` | `src/llama-kv-cache-iswa.h:17` | La caché KV de un modelo con ventana deslizante intercalada: una caché para cada tipo de capa |
| `build_attn_inp_kv_iswa`, `build_amax_src1_f16` | `src/llama-graph.cpp:2058`, `:1178` | Las entradas de la atención con esa caché; el reescalado de la activación de los expertos de Flint |
| `rpc_node_ok`, `rpc_serve_auth` | `ggml/src/ggml-rpc/ggml-rpc.cpp:2566`, `:3714` | Lo que calcula un servidor RPC; su autenticación |
| `fHttpHostAllowed` | `tools/server/server-http.cpp:178` | La comprobación de la cabecera `Host` de `llama-server` |

## 4. Flujos principales

### Cargar un modelo

1. `ggml_backend_xdna_reg` registra el dispositivo; `fProbe` lee los ajustes y los manifiestos y comprueba la NPU (`XdnaDevice::mQueryInfo`).
2. llama.cpp da al tipo de búfer del backend los pesos de las multiplicaciones que este toma; con `mmap`, `fDeviceBufferFromHostPtr` crea esos búferes sobre la proyección del archivo.
3. `fDeviceInit` abre el runner (`fOpenRunner` → `XdnaRunner::mOpen`: tres contextos `vContext`, `vGemvContext` y `vMoeContext` compartiendo dispositivo y heap). Los estados recurrentes van a búferes del controlador (`fStateBufferTypeAlloc`).

### Un lote de un prompt

1. El planificador da al backend sus partes del grafo; `fGraphCompute` las recorre.
2. Una `RMS_NORM` y su `MUL` antes de unas multiplicaciones: sus escalas se calculan en ese momento (`fRowScales`), y las filas se normalizan al convertirlas.
3. Varios `MUL_MAT` de las mismas filas: `fMulMat` → `fMatrixOf` (los pesos se convierten la primera vez, `fConvert`) → `fMultiply`: las filas en un búfer de A (`mPackRows`), la orden (`mSubmitPacked`) y C leído en el resultado (`mFinishPacked` → `fXdnaGemmUnpackC`), con la suma residual o el `CONCAT` cuando van detrás.
4. La puerta y la subida seguidas de una SwiGLU: `fMulMatGlu`, con la SwiGLU en las filas de `ffn_down` (`mUnpackSwigluRows`).
5. Atención lineal: la convolución corta con su SiLU al leer la C de `attn_qkv` (`fConvFold` → `fConvStrips`; en el anfitrión cuando no puede ir con ella, `fConvOnHost`), la recurrencia (`fGatedDeltaNet` → `mRunGdn`, con las normas L2 de q y k dentro) y la norma con puerta en las filas de `ssm_out` (`fGatedNorm`).
6. Atención: `fFlashAttention` (flujo siguiente).

### Generar un token

1. `fGraphCompute` encuentra un `MUL_MAT` de una fila que empieza un bloque feed-forward denso (`fFeedForward` reconoce `MUL_MAT` → `ADD` → `RMS_NORM` → `MUL` → 2 `MUL_MAT` → `GLU` → `MUL_MAT` → `ADD` y, detrás, la norma y las proyecciones de la capa siguiente, o — también a través de un `GET_ROWS` de 1 fila y de una vista prefijo `VIEW` `head_part` de desplazamiento 0 (`cWeightViewKeyTag`) en los grafos MTP — la norma de salida y los logits) o, cuando los expertos enrutados de 1 token corren en el procesador, un bloque de experto compartido (`fSharedExpertChain` reconoce `{gate_shexp, up_shexp} + GLU + down_shexp [+ puerta] + ADD + ADD` más la norma y las proyecciones de la capa siguiente).
2. Las matrices del overlay encadenado (`fGemvMatrixOf` → `mLoadChainMatrix`), las fases (`XdnaChainPhase`), una orden (`mRunChain` → `fXdnaChainPlan`) y los resultados leídos donde los quiere el grafo (`fXdnaChainReadY`).
3. Lo que no forma un bloque así va por `fMultiplyRows` → `mChainMultiplyRows`, o por `xdna2-moe` (`fEnsureMoeDenseGroup`, `fRunDenseMoeFfn`) cuando los expertos enrutados corren allí de 1 a 8 filas para no cambiar nunca de overlay.

### La atención de un lote

`fFlashAttention`: por cada cabeza de claves, K y los trozos de V (4.096 claves como mucho) se convierten desde la caché en F16 (`mLoadAttnB`); por cada bloque de filas (las consultas de las cabezas que la comparten, por tokens), S = Q·Kᵀ en una orden (`mPackAttnQ`, `mSubmitPacked`), P = exp(S + máscara − máximo) en las filas de los trozos de P·V (`mPackAttnP` → `fXdnaGemmPackAttnP`, sumando también exp(sumidero − máximo) en cada fila cuando la capa lleva sumideros de atención), O = P·V con todos sus trozos en vuelo, y O dividido entre las sumas de las filas de P, en el resultado. Las cabezas van una tras otra, con la NPU y el anfitrión solapados. Sin norma de q y k (`fNormedQueries`), Q y K van en dos trozos de bfp16.

### La recurrencia de un lote

`fGatedDeltaNet` → `XdnaRunner::mRunGdn`: `fXdnaGdnPack` escribe los estados y los trozos de 16 tokens de una parte de las rondas mientras la NPU calcula las partes anteriores (hasta 4 órdenes en vuelo), y `fXdnaGdnUnpack` lee las salidas y los estados finales.

### Expertos enrutados (`MUL_MAT_ID`)

`fRunMoeStep`: agrupa los pares `(token, ranura)` por experto en trozos de 1 a 24 filas (`fBuildMoeRouting`), pliega cualquier experto compartido contiguo (`ffn_*_shexp` y su puerta escalar `ffn_gate_inp_shexp`) como 1 o 2 expertos virtuales (`fMatchMoeSharedExpert`, `fEnsureMoeWithShared` → `mLoadMoeMatrix`, liberando las páginas de solo lectura de los expertos 3D con `MADV_DONTNEED` cuando `GGML_XDNA_MOE_MIN_TOKENS` vale 1), ejecuta `{gate_exps, up_exps} + SwiGLU` en una orden (`mRunMoe`), absorbe el reescalado opcional `build_amax_src1_f16` de 10 nodos de Flint (`fMatchMoeAmax`) para continuar directamente en `down_exps` desde la Y conservada (`mRunMoeFromKept` → `fXdnaMoeTranscodeYToX`) y escribe la reducción ponderada por token más la `ADD` residual mientras lee Y (`fMatchMoeTail`, `fXdnaMoeReadY`).

### Síntesis de voz (`Qwen3-TTS`)

`llama-tts` (`tools/tts/tts.cpp`): en los modelos `Base`, `mtmd_helper_encode_speech` decodifica la voz de referencia (`miniaudio`), extrae características mel Slaney y ejecuta `ECAPA-TDNN` (`qwen3tts-spkenc.cpp`) en `F32` en el procesador, mientras que `CustomVoice` busca un vector `<|spk_*|>` integrado (`mtmd_gen_audio_get_voice`, `--tts-voice`) y `VoiceDesign` / `1.7B-CustomVoice` anteponen `<|im_start|>user\n{instruct}<|im_end|>\n` (`--tts-instruct`); en cada fotograma de 12,5 Hz, `llama_decode` ejecuta el Talker de 28 capas (`src/models/qwen3tts.cpp`) y `mtmd_generate_audio_tokens` ejecuta 16 pasos de 1 token del transformer `code_gen` de 5 capas (`gen_code_head_2d[0..14]`), ambos encadenados en `xdna2-gemv-chain` con 0 cambios de overlay; cada 72 fotogramas, `mtmd_decode_audio_codes` cambia una sola vez a `xdna2-gemm-bfp16` para ejecutar `code2wav` (`conv1d_w_2d`, `conv1d_tr_w_2d`) y generar PCM a 24 kHz.

## 5. Mapa de entradas

| Orden | Manejador | Archivo |
|---|---|---|
| `llama-cli`, `llama-server`, `llama-tts`, `llama-bench`, `llama-perplexity`, `llama-quantize`, `llama-completion`… | Los de llama.cpp | `tools/*/` |
| `ggml-rpc-server` | `ggml_backend_rpc_start_server` | `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| `llama-xdna-info [--check-buffers]` | `main` | `tools/xdna-info/xdna-info.cpp` |
| `llama-xdna-run --pdi --insts --in --out [--repeat]` | `main` | `tools/xdna-run/xdna-run.cpp` |
| `llama-xdna-insts gemm \| gemv \| chain \| moe \| gdn \| empty \| dump …` | `main` | `tools/xdna-insts/xdna-insts.cpp` |
| `llama-xdna-gemm [pack \| unpack \| gdn] …` | `main` | `tools/xdna-gemm/xdna-gemm.cpp` |
| `llama-xdna-gemv [--ffn \| --gdn \| --la \| --moe] …` | `main` | `tools/xdna-gemv/xdna-gemv.cpp` |
| `llama-xdna-probe chain \| switch \| units` | `main` | `tools/xdna-probe/xdna-probe.cpp` |
| `build-overlays.sh <entorno virtual> <carpeta>` | `fMain` | `ggml/src/ggml-xdna/overlays/build-overlays.sh` |
| `install-update-reinstall-debian.sh`, `-alpine.sh` | `fMain` | `deploy/` |

Las variables de entorno del backend están en [MANUAL.es-ES.md](MANUAL.es-ES.md#8-ajustes-del-backend-de-la-npu); se leen en `fProbe`.

## 6. Análisis de impacto

- **Un overlay** (`overlays/designs/`, `kernels/`): el anfitrión escribe sus flujos a partir de su manifiesto; una disposición o unos parámetros nuevos obligan a cambiar los generadores de flujos (`fXdnaGemmPlan`, `fXdnaGdnPlan`, `fXdnaChainPlan`, `fXdnaMoePlan`) y, en el overlay encadenado o el de expertos, a una versión nueva de su manifiesto (`xdna2-gemv <n>`, `xdna2-moe <n>`), que el anfitrión comprueba (`fXdnaChainKey` resume solo la geometría y los tipos de operación, nunca escalares de ejecución como `vEps`, para reutilizar los búferes de instrucciones sin llenar los 64 MiB de `DEV_HEAP`). Tanto `xdna2-gemv-chain` (`vGemvContext`) como `xdna2-moe` (`vMoeContext`) tienen que seguir en sus propios `XdnaContext`: compartir `vContext` con `xdna2-gemm-bfp16` acababa en `timeout` tras órdenes en `Q5_K`/`Q6_K`. Un overlay por defecto nuevo hay que copiarlo en `ggml/src/ggml-xdna/overlays/`. Un diseño nuevo se ejecuta primero una sola vez: una orden que se cuelga cuesta apagar el ordenador.
- **El codificador** (`xdna-bfp16.cpp`): todos los resultados de la NPU dependen de él; un cambio tiene que dar los mismos bytes (comparar byte a byte los logits de un modelo antes y después).
- **Las fusiones de `fGraphCompute`**: dependen del orden de los nodos que construye `src/models/qwen35.cpp` y de las partes que hace el planificador. Un cambio del grafo en upstream puede desactivar una fusión sin ningún error: el informe del backend (`-lv 4`) cuenta cada fusión.
- **Escribir un resultado antes de su turno** (`fConvFold`): la `SILU` de la convolución corta se escribe al leer la C de la proyección, antes de su turno en el grafo, y eso solo es seguro porque se comprueba que su memoria no se solapa con nada que lean o escriban los nodos de en medio; una fusión nueva que escriba un resultado antes de tiempo necesita la misma comprobación. Depende además de que `src/models/qwen35.cpp` meta la convolución en el grafo antes que el estado recurrente.
- **`graph_get_cb`**: da operaciones al acelerador; las sondas de `resolve_fused_ops` construyen flash attention con 1 token y la recurrencia con 16, que el backend tiene que seguir rechazando, porque si no llama.cpp desactiva esas operaciones para todo el modelo.
- **Los búferes del runner** (`cXdnaRunnerSlotsA`, `cXdnaRunnerSlotsC`): la SwiGLU en `ffn_down`, la norma con puerta y la atención usan ranuras fijas; una orden en vuelo es dueña de sus ranuras de A y de C hasta que termina.
- **Las funciones con varias versiones** (`XDNA_CLONES`, en `ggml-xdna.cpp`): se compilan para AVX-512, AVX2 y x86-64 básico, y glibc elige la versión al arrancar el programa con una función indirecta (ifunc). musl (Alpine) no tiene ifunc, así que allí solo existe la versión básica; hoy solo la usa la emulación (`fDot`).
- **Las operaciones del procesador** (`ggml/src/ggml-cpu/ops.cpp`): los cambios dan los mismos resultados bit a bit; tras cualquier cambio, comparar byte a byte los logits del procesador solo antes y después (`GGML_XDNA_DISABLE=1 llama-perplexity -c 512 -b 512 --chunks 8 --kl-divergence-base <archivo>` y luego `cmp`).
- **El protocolo RPC**: las órdenes propias de este fork van detrás de las de upstream; al portar órdenes nuevas de upstream, las propias cambian de número, y los clientes y servidores de antes y de después dejan de entenderse.

## 7. Puntos de extensión

- **Una operación para el backend**: una función que diga cuándo se toma (como `fAttnTaken`), su caso en `fDeviceSupportsOp`, su manejador en `fGraphCompute`, sus contadores en `XdnaTotals` y en el informe de `fBackendFree`, y un interruptor `GGML_XDNA_NO_…` en `XdnaSettings`, leído en `fProbe` (`fConvTaken`, `fConvFold` y `fConvOnHost` son un ejemplo pequeño: una operación que se toma, su fusión y su versión en el anfitrión). Si el planificador no se la diera al backend, puede dársela `graph_get_cb`, en `src/llama-context.cpp`.
- **Un overlay**: su diseño en `overlays/designs/` y sus kernels en `kernels/`, su receta en `overlays/build-overlays.sh` (CMake lo compila y lo copia con `-DGGML_XDNA_BUILD_OVERLAYS=ON -DGGML_XDNA_MLIR_AIE_VENV=<entorno virtual>`), su manifiesto leído en `xdna-overlay.cpp`, `xdna-chain.cpp` o `xdna-moe.cpp`, su flujo escrito por un plan y un método del runner que lo envíe.
- **Una arquitectura de modelo**: se porta desde upstream (el modelo en `src/models/`, `src/llama-arch.*`, `src/llama-model.*`, `src/llama-hparams.h`, `src/llama-model-saver.cpp` y su modelo de juguete en `tests/test-llama-archs.cpp`); sus operaciones que no estén en `rpc_node_ok` no las calcula un servidor RPC hasta que se añadan allí; las fusiones del backend reconocen los patrones de `qwen35`, y una arquitectura nueva necesita los suyos.
- **Un ajuste**: un campo de `XdnaSettings`, leído en `fProbe`, y su fila en la tabla de [MANUAL.es-ES.md](MANUAL.es-ES.md#8-ajustes-del-backend-de-la-npu) y en sus traducciones.
