ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/slotcontrast

# Keep the PPU-specific torch and torchvision supplied by the base image.
RUN python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "numpy==1.26.4" \
    "torchmetrics==1.2.0" \
    "pytorch-lightning==1.9.5" \
    "timm==1.0.26" \
    "omegaconf==2.3.0" \
    "webdataset==0.2.35" \
    "einops==0.8.2" \
    "scipy==1.15.3" \
    "scikit-image==0.25.0" \
    "moviepy==1.0.3" \
    "av==12.0.0" \
    "comet-ml==3.57.3" \
    "requests==2.32.5"

COPY slotcontrast /opt/slotcontrast/slotcontrast
COPY ppu/homegrid_7slots_ss.yaml /opt/slotcontrast/configs/homegrid_7slots_ss.yaml

WORKDIR /opt/slotcontrast

RUN python3 -c "import torch, torchvision; assert torch.__version__.startswith('2.10.0'), torch.__version__; print('torch', torch.__version__, 'torchvision', torchvision.__version__)" \
 && python3 -c "import slotcontrast.train; print('SlotContrast import OK')"

CMD ["python3", "-m", "slotcontrast.train", "--help"]
