rm -rf /etc/pip.conf
rm -rf /etc/xdg/pip/pip.conf
rm -rf /root/.config/pip/pip.conf
rm -rf /root/.pip/pip.conf
rm -rf /usr/pip.conf
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install transformers==4.48.0 pytrie sentencepiece ring_flash_attn
# cp -r BMTrain/ /root/BMTrain
# yes|pip uninstall bmtrain && source ~/.bashrc && cd /root/BMTrain && pip install .
# cd /ml-cross-entropy && rm -rf ./build && pip install .
# yes|pip uninstall burst-attn 
# cd /Burst-Attention && pip install .
# pip install BMTrain/bmtrain-1.0.0-cp310-cp310-linux_x86_64.whl

