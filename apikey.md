sk-ws-H.PHMPPRX.3yr9.MEUCIGEyACTdWACEfeeSgcXrt7jdvCqYBGeTbsEmm3IbxX8zAiEA0xCMaC4lzHkKqRgYvBoPTgMDO_OqdembTe_tJCKYT7I

cd graspnet_orbbec
source scripts/activate.sh
export DASHSCOPE_API_KEY=sk-ws-H.PHMPPRX.3yr9.MEUCIGEyACTdWACEfeeSgcXrt7jdvCqYBGeTbsEmm3IbxX8zAiEA0xCMaC4lzHkKqRgYvBoPTgMDO_OqdembTe_tJCKYT7I   # 不要写进仓库
python vlm/run_vlm.py --image vlm/testdata/01_floor.png --target 蓝色湿巾包
python vlm/run_vlm.py --eval 
