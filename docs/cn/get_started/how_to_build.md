# How To Build

### Prerequisites (one-time setup)
```bash
apt install libboost-all-dev
script/download_third_party_lib.sh
script/install_zmq_ubuntu.sh
script/install_grpc_ubuntu.sh
```

```bash
pip install scikit-build-core torch==2.8.0 cmake==4.3.2
```

### Editable install (development)
```bash
pip install -v --no-build-isolation -e .
```

### Build wheel
```bash
pip wheel -v --no-build-isolation -w dist/ .
# cp38-abi3-manylinux_2_35_x86_64
auditwheel repair -w dist/ dist/dtorch-*linux_x86_64.whl
```
