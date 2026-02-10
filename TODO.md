# TODO

## Investigate SVO recording encoder initialization failure (NVENC)

### Summary
When recording point clouds as SVO (`--enable-point-cloud --record-pointcloud-format=svo`), the app fails to start SVO recording due to NVENC initialization errors. The recording continues without SVO output.

### Repro
```bash
./build/zed_app --record --record-duration=15 --enable-point-cloud --record-pointcloud-format=svo
```

### Observed logs
```
[ZED][SLHW] Failed to create encoder. Err : CreateEncoder : m_nvenc.nvEncInitializeEncoder(m_hEncoder, &m_initializeParams) returned error 8 ...
[WARN] Failed to start SVO recording.
[INFO] Recording started.
```

### Environment snapshot
- GPU: NVIDIA GeForce GTX 1660 Ti (TU116)
- Driver: 590.48.01 (open kernel module)
- `nvidia-smi`: `Failed to initialize NVML: Unknown Error`
- NVENC runtime: `/lib/x86_64-linux-gnu/libnvidia-encode.so` present
- Device nodes: `/dev/nvidia-caps/*` are root-only (`cr--------`)
- `nvidia-compute-utils-590` not installed

3. Re-check `/dev/nvidia-caps/*` permissions; ensure `video` or `render` group has access.
4. Add a CLI/config option to force lossless SVO to bypass NVENC if needed.
