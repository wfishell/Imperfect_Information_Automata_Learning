# ---------- Base: PyTorch with CUDA + devel tools ----------
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    cmake bison flex pkg-config g++ make \
    libgmp-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- Build Spot (CLI tools only, no Python bindings) ----------
FROM base AS spot-builder
WORKDIR /src

ENV SPOT_VERSION=2.14.1

RUN wget -q http://www.lre.epita.fr/dload/spot/spot-${SPOT_VERSION}.tar.gz \
    && tar xzf spot-${SPOT_VERSION}.tar.gz \
    && cd spot-${SPOT_VERSION} \
    && ./configure --prefix=/usr/local --disable-python --enable-tools \
    && make -j"$(nproc)" \
    && make install

# ---------- Build SyFCo (same base to match architecture) ----------
FROM base AS syfco-builder
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    ghc cabal-install \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/reactive-systems/syfco.git \
    && cd syfco \
    && cabal update \
    && cabal v2-install --installdir=/usr/local/bin --overwrite-policy=always

# ---------- Final image ----------
FROM base

# Copy Spot binaries and libs
COPY --from=spot-builder /usr/local /usr/local
RUN ldconfig

# Copy SyFCo binary
COPY --from=syfco-builder /usr/local/bin/syfco /usr/local/bin/syfco

# Install GMP runtime (needed by syfco at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgmp10 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workdir

COPY . /workdir/Automata_SSM_Learning

RUN git clone https://github.com/eric-hsiung/REMAP.git /workdir/REMAP

RUN pip install --no-cache-dir \
    jinja2 sympy networkx setuptools typing-extensions \
    torchvision torchaudio torchtext torchmetrics \
    anthropic \
    aalpy==1.5.1 \
    z3-solver

ENV PYTHONPATH="/workdir/REMAP"

CMD ["/bin/bash"]
