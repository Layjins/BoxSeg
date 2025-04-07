work_root=/path/workspace/research/segmentation/BoxSeg
#abs_path=$(cd $(dirname $0); pwd)
#export PYTHONPATH=".":"..":$PYTHONPATH
#cd "$abs_path/.."
cd ${work_root}
# train
python3 setup.py build develop
cd AdelaiDet
python3 setup.py build develop
cd ..
python3 projects/BoxSeg/train_net.py \
   --config-file projects/BoxSeg/configs/coco/boxseg_r50_1x.yaml \
   --num-gpus 4  \
   SOLVER.IMS_PER_BATCH 16 \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r50_1x #\
   #MODEL.WEIGHTS /path/workspace/research/segmentation/BoxSeg_checkpoints/res50_1x/model_final.pth


# test
cd ${work_root}
python3 setup.py build develop
cd ${work_root}/AdelaiDet
python3 setup.py build develop
cd ..
python3 ${work_root}/projects/BoxSeg/train_net.py \
   --config-file ${work_root}/projects/BoxSeg/configs/coco/boxseg_r50_1x.yaml \
   --eval-only \
   --num-gpus 4  \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r50_1x/test-dev \
   MODEL.WEIGHTS ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r50_1x/model_final.pth


# train
python3 setup.py build develop
cd AdelaiDet
python3 setup.py build develop
cd ..
python3 projects/BoxSeg/train_net.py \
   --config-file projects/BoxSeg/configs/coco/boxseg_r50_3x.yaml \
   --num-gpus 4  \
   SOLVER.IMS_PER_BATCH 16 \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r50_3x #\
   #MODEL.WEIGHTS /path/workspace/research/segmentation/BoxSeg_checkpoints/res50_1x/model_final.pth


# test
cd ${work_root}
python3 setup.py build develop
cd ${work_root}/AdelaiDet
python3 setup.py build develop
cd ..
python3 ${work_root}/projects/BoxSeg/train_net.py \
   --config-file ${work_root}/projects/BoxSeg/configs/coco/boxseg_r50_3x.yaml \
   --eval-only \
   --num-gpus 4  \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r50_3x/test-dev \
   MODEL.WEIGHTS ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r50_3x/model_final.pth



# train
python3 setup.py build develop
cd AdelaiDet
python3 setup.py build develop
cd ..
python3 projects/BoxSeg/train_net.py \
   --config-file projects/BoxSeg/configs/coco/boxseg_r101_3x.yaml \
   --num-gpus 4  \
   SOLVER.IMS_PER_BATCH 16 \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r101_3x #\
   #MODEL.WEIGHTS /path/workspace/research/segmentation/BoxSeg_checkpoints/res50_1x/model_final.pth


# test
cd ${work_root}
python3 setup.py build develop
cd ${work_root}/AdelaiDet
python3 setup.py build develop
cd ..
python3 ${work_root}/projects/BoxSeg/train_net.py \
   --config-file ${work_root}/projects/BoxSeg/configs/coco/boxseg_r101_3x.yaml \
   --eval-only \
   --num-gpus 4  \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r101_3x/test-dev \
   MODEL.WEIGHTS ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r101_3x/model_final.pth



# train
python3 setup.py build develop
cd AdelaiDet
python3 setup.py build develop
cd ..
python3 projects/BoxSeg/train_net.py \
   --config-file projects/BoxSeg/configs/coco/boxseg_r101_dcn_3x.yaml \
   --num-gpus 4  \
   SOLVER.IMS_PER_BATCH 16 \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r101_dcn_3x #\
   #MODEL.WEIGHTS /path/workspace/research/segmentation/BoxSeg_checkpoints/res50_1x/model_final.pth


# test
cd ${work_root}
python3 setup.py build develop
cd ${work_root}/AdelaiDet
python3 setup.py build develop
cd ..
python3 ${work_root}/projects/BoxSeg/train_net.py \
   --config-file ${work_root}/projects/BoxSeg/configs/coco/boxseg_r101_dcn_3x.yaml \
   --eval-only \
   --num-gpus 4  \
   OUTPUT_DIR ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r101_dcn_3x/test-dev \
   MODEL.WEIGHTS ${work_root}/../BoxSeg_checkpoints/gavyn.boxseg_r101_dcn_3x/model_final.pth
