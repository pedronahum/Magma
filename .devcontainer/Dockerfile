# SwiftTensor Development Container
# Based on Swift official image with XLA dependencies

FROM swift:5.9-jammy

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    libcurl4-openssl-dev \
    libssl-dev \
    zlib1g-dev \
    lldb \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set up XLA directory (users will mount or download XLA here)
ENV SWIFTTENSOR_XLA_PATH=/opt/xla
RUN mkdir -p /opt/xla && chown $USERNAME:$USERNAME /opt/xla

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Verify Swift installation
RUN swift --version
