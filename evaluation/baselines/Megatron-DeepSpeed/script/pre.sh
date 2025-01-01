rm -rf /etc/pip.conf
rm -rf /etc/xdg/pip/pip.conf
rm -rf /root/.config/pip/pip.conf
rm -rf /root/.pip/pip.conf
rm -rf /usr/pip.conf
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install transformers sentencepiece deepspeed==0.15.0
