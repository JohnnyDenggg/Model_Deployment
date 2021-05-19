# 在python基础上构建
FROM python:3.7-stretch
# 在镜像系统中/tmp目录下操作
WORKDIR /tmp
# 将主机当前目录下的txt文件复制到镜像系统/tmp目录下
COPY ./requirements.txt .
# 运行安装依赖包
RUN pip install -r requirements.txt
# 暴露5000端口
EXPOSE 5000
#将工作目录切换到/app下
WORKDIR /app
# 将主机当前目录下start.sh文件复制到镜像/home目录下
COPY ./start.sh /home/start.sh
# 镜像在容器启动时执行命令，虚拟机开启后自动运行脚本
CMD ["/home/start.sh"]
