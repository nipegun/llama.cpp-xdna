# uapi

`amdxdna_accel.h` is the user-space interface of the `amdxdna` driver of Linux (the ioctls of `/dev/accel/accel*`), copied without changes from `include/uapi/drm/amdxdna_accel.h` of Linux 7.3.4. Its license is GPL-2.0 WITH Linux-syscall-note, which lets user programs use it.

It is here because Debian 13 ships the user-space headers of Linux 6.12, which do not have it yet: the driver entered the kernel in Linux 6.14. It includes `drm.h`, which comes from the system (`linux-libc-dev`).

To update it, copy the file from a newer kernel. Do not edit it.
