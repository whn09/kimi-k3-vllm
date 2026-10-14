# Kimi-K3 on vLLM + DeepEP v2 over EFA, for p6-b300 (sm_103).
#
#   docker build -t kimi-k3-vllm:<tag> .        # or: bash 01_build_push.sh
#
# What the stock vllm/vllm-openai nightly lacks for PD on EFA with deepep_v2:
#   1. EFA userspace (libfabric + aws-ofi-nccl with the GIN plugin). Without it
#      NCCL falls back to sockets across nodes and NIXL's LIBFABRIC plugin cannot
#      even load (it dlopen()s libfabric.so.1, which the nightly does not ship).
#   2. NCCL >= 2.31. The nightly pins nvidia-nccl-cu13 2.30.7, on which
#      NCCL_GIN_TYPE=5 (EFA-GDA) cannot initialise at all.
#   3. A DeepEP built for sm_103 against that NCCL. The bundled deep_ep 2.0.0 is
#      upstream d4f41e4 compiled for '9.0a 10.0a' against NCCL 2.30.7, i.e. no
#      B300 cubins and the wrong NCCL; we replace it with amazon-contributing
#      DeepEP (same ElasticBuffer API vLLM calls, plus the EFA fixes).
#   4. vllm-router (the PD proxy the recipe uses).

# PINNED, not :latest/:nightly -- a floating tag makes "the image we measured"
# unanswerable. This sha has vllm/models/kimi_k3 and --all2all-backend deepep_v2.
# cu130 (CUDA 13.0) rather than cu134: the hosts' driver is 595.x.
ARG VLLM_BASE=vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989
FROM ${VLLM_BASE}

USER root
ENV DEBIAN_FRONTEND=noninteractive

# libhwloc15: NIXL's libplugin_LIBFABRIC.so links libhwloc.so.15, which the
# nightly does not have either. libdw-dev: DeepEP's JIT.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential wget curl patch pciutils \
        autoconf automake libtool libhwloc15 libdw-dev \
    && rm -rf /var/lib/apt/lists/*

ARG GDRCOPY_VERSION=2.5.2
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz && \
    tar xf v${GDRCOPY_VERSION}.tar.gz && cd gdrcopy-${GDRCOPY_VERSION} && \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local && \
    rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# EFA userspace, pinned. 1.50.0's aws-ofi-nccl (1.21.x) exports ncclGinPlugin_v14
# (EFA-GDA). Do not move to NCCL 2.32.x without moving aws-ofi-nccl to master:
# 2.32 asks the GIN plugin for backendVersion 2, which 1.21.x does not speak.
ARG EFA_INSTALLER_VERSION=1.50.0
ENV K3_EFA_INSTALLER=${EFA_INSTALLER_VERSION}
RUN set -eu; \
    cd /tmp; \
    curl -fSL --retry 3 -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz; \
    tar xzf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz; \
    grep -q "^## \[${EFA_INSTALLER_VERSION}\]" aws-efa-installer/ChangeLog.md \
      || { echo "FATAL: tarball is not EFA installer ${EFA_INSTALLER_VERSION}"; exit 1; }; \
    cd aws-efa-installer; \
    (apt-get install -y ./DEBS/UBUNTU2404/x86_64/rdma-core/*.deb || true); \
    ./efa_installer.sh -y --skip-kmod -g --no-verify --skip-rdma-core; \
    ldconfig; \
    PLUGIN=$(find /opt/amazon -name 'libnccl-net*.so*' -type f 2>/dev/null | head -1); \
    test -n "$PLUGIN" || { echo "FATAL: no libnccl-net*.so under /opt/amazon"; exit 1; }; \
    nm -D "$PLUGIN" | grep -qw ncclGinPlugin_v14 \
      || { echo "FATAL: $PLUGIN exports no ncclGinPlugin_v14 => GIN would silently use the type-2 CPU proxy"; exit 1; }; \
    echo "EFA ${EFA_INSTALLER_VERSION}: $PLUGIN has ncclGinPlugin_v14"; \
    rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV NCCL_NET_PLUGIN=ofi
ENV PATH="/opt/amazon/efa/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/lib:${LD_LIBRARY_PATH}"

# NCCL >= 2.31 BEFORE DeepEP, so DeepEP links the NCCL it will run with
# (DeepEP's check_nccl_so() compares the loaded and linked libnccl byte-for-byte).
# The nightly also pins nccl through UV_OVERRIDE=/etc/uv-overrides.txt; rewrite it
# so a later `uv pip install` cannot silently put 2.30.7 back.
ARG NCCL_PIP_VER=2.31.2
RUN set -eu; \
    pip install --no-cache-dir --no-deps --upgrade "nvidia-nccl-cu13==${NCCL_PIP_VER}"; \
    echo "nvidia-nccl-cu13==${NCCL_PIP_VER}" > /etc/uv-overrides.txt; \
    v=$(python3 -c "import importlib.metadata as m; print(m.version('nvidia-nccl-cu13'))"); \
    test "$v" = "${NCCL_PIP_VER}" || { echo "FATAL: pip NCCL is $v, wanted ${NCCL_PIP_VER}"; exit 1; }; \
    echo "pip NCCL $v"

# amazon-contributing/DeepEP main @ 6de427b: the ref the official sm103 EP image
# on these hosts was built from. 10.3 = sm_103 (B300); 10.0 kept so the same
# image also runs on b200.
ARG DEEPEP_REF=6de427b1728aaac46f473f91b429397eb24de691
ARG DEEPEP_ARCH="10.0;10.3"
ENV K3_DEEPEP_REF=${DEEPEP_REF}
ENV EP_JIT_CACHE_DIR=/root/.deep_ep
RUN set -eu; \
    pip uninstall -y deep_ep deep-ep 2>/dev/null || true; \
    SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); \
    rm -rf "$SP/deep_ep" "$SP"/deep_ep-*; \
    mkdir -p /opt/DeepEP; cd /opt/DeepEP; \
    git init -q; \
    git remote add origin https://github.com/amazon-contributing/DeepEP.git; \
    git fetch -q --depth 1 origin "$DEEPEP_REF"; \
    git checkout -q FETCH_HEAD; \
    git submodule update -q --init --recursive --depth 1; \
    NCCL_LIB="$(python3 -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib"))')"; \
    TORCH_CUDA_ARCH_LIST="$DEEPEP_ARCH" LIBRARY_PATH="$NCCL_LIB:${LIBRARY_PATH:-}" \
      python3 setup.py bdist_wheel; \
    pip install --no-cache-dir --no-deps dist/*.whl; \
    rm -rf build dist; \
    git rev-parse HEAD > /opt/DeepEP/BUILD_REF; \
    n=$(python3 -c "\
from importlib.metadata import distributions;\
print(len({d.metadata['Name'] for d in distributions() \
  if any(str(f).startswith('deep_ep/') for f in (d.files or []))}))"); \
    test "$n" -eq 1 || { echo "FATAL: $n distributions own deep_ep/, expected 1"; exit 1; }

ARG VLLM_ROUTER_VER=
RUN pip install --no-cache-dir "vllm-router${VLLM_ROUTER_VER:+==$VLLM_ROUTER_VER}" && \
    which vllm-router

# Preflight. No GPU during `docker build`, so anything that dlopen()s libcuda
# runs with the CUDA stubs on the path.
RUN set -eu; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"; \
    P=$(python3 -c "import nixl_cu13, os; print(os.path.dirname(nixl_cu13.__file__))")/../nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so; \
    test -f "$P" || { echo "FATAL: NIXL LIBFABRIC plugin missing at $P"; exit 1; }; \
    ldd "$P" | grep -E 'libfabric|hwloc'; \
    ldd "$P" | grep 'libfabric.so.1 => /opt/amazon/efa/lib' \
      || { echo "FATAL: NIXL LIBFABRIC plugin does not resolve the EFA libfabric"; exit 1; }; \
    ! ldd "$P" | grep -E '(libfabric|hwloc).*not found' \
      || { echo "FATAL: NIXL LIBFABRIC plugin has unresolved deps"; exit 1; }; \
    python3 -c "import deep_ep; assert hasattr(deep_ep, 'ElasticBuffer'); print('deep_ep', deep_ep.__version__, deep_ep.__file__)"; \
    python3 -c "from vllm.utils.import_utils import deep_ep_v2_unavailable_reason as r; x = r(); print('vllm deepep_v2:', x or 'available'); assert x is None" \
      || echo "WARN: deepep_v2 availability probe failed under the libcuda stub (it reads the runtime NCCL version); re-checked at launch"; \
    python3 -c "import vllm.models.kimi_k3; print('vllm kimi_k3 OK')"; \
    NV=$(python3 -c "import importlib.metadata as m; print(m.version('nvidia-nccl-cu13'))"); \
    python3 -c "import sys; v=tuple(int(x) for x in '$NV'.split('.')[:2]); sys.exit(0 if v >= (2,31) else 1)" \
      || { echo "FATAL: a later layer downgraded NCCL to $NV"; exit 1; }; \
    echo "preflight OK: DeepEP $(cat /opt/DeepEP/BUILD_REF) arch ${DEEPEP_ARCH} nccl $NV efa ${K3_EFA_INSTALLER}"

WORKDIR /workspace
ENTRYPOINT []
