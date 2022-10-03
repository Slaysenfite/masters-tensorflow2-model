#!/bin/bash
#
#cd $HOME/data
#
#git clone https://github.com/Slaysenfite/ddsm_lr

cd $HOME/masters-tensorflow2-model/

git checkout instance/uber

rm -r output

workon wesselsenv

#python python_src/DdsmMetadataGenerator.py

# Run Scripts
#python python_src/ResNetMain.py --optimizer=adam --id=CRD-I --dataset=cbis --preloaded_weights=True --epochs=100
#python python_src/NewResNetMain.py --id=CRD-GA --optimizer=adam --meta_heuristic=ga --preloaded_experiment=MCRD001-ImageNet
#python python_src/NewResNetMain.py --id=CRD-PSO --optimizer=adam --meta_heuristic=pso --preloaded_experiment=MCRD001-ImageNet

#python python_src/XceptionNetMain.py --optimizer=adam --id=CXD1-I --dataset=cbis --preloaded_weights=True --epochs=100
#python python_src/NewXceptionNetMain.py --id=CXD-GA --optimizer=adam --meta_heuristic=ga--preloaded_experiment=MCXD001-ImageNet
#python python_src/NewXceptionNetMain.py --id=CXD-PSO --optimizer=adam --meta_heuristic=pso --preloaded_experiment=MCXD001-ImageNet

#python python_src/ResNetMain.py --optimizer=adam --id=CRF-I --dataset=ddsm --preloaded_weights=True --epochs=100
#python python_src/NewResNetMain.py --id=CRF-PSO --optimizer=adam --preloaded_experiment=CRF-I --meta_heuristic=pso --dataset=ddsm
#python python_src/NewResNetMain.py --id=CRF-GA --optimizer=adam --preloaded_experiment=CRF-I --meta_heuristic=ga --dataset=ddsm
#
#python python_src/XceptionNetMain.py --id=CXF-I --optimizer=adam  --dataset=ddsm --preloaded_weights=True --epochs=100
#python python_src/NewXceptionNetMain.py --id=CXF-PSO --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=pso --dataset=ddsm
#python python_src/NewXceptionNetMain.py --id=CXF-GA --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=ga --dataset=ddsm

python3 python_src/NewResNetMain.py --id=CRD-GA1 --optimizer=adam --meta_heuristic=ga --preloaded_experiment=CRD-I
python3 python_src/NewResNetMain.py --id=CRD-GA2 --optimizer=adam --meta_heuristic=ga --preloaded_experiment=CRD-I
python3 python_src/NewResNetMain.py --id=CRD-GA3 --optimizer=adam --meta_heuristic=ga --preloaded_experiment=CRD-I
python3 python_src/NewResNetMain.py --id=CRD-GA4 --optimizer=adam --meta_heuristic=ga --preloaded_experiment=CRD-I
python3 python_src/NewResNetMain.py --id=CRD-GA5 --optimizer=adam --meta_heuristic=ga --preloaded_experiment=CRD-I

python python_src/NewXceptionNetMain.py --id=CXF-GA1 --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=ga
python python_src/NewXceptionNetMain.py --id=CXF-GA2 --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=ga
python python_src/NewXceptionNetMain.py --id=CXF-GA3 --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=ga
python python_src/NewXceptionNetMain.py --id=CXF-GA4 --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=ga
python python_src/NewXceptionNetMain.py --id=CXF-GA5 --optimizer=adam --preloaded_experiment=CXF-I --meta_heuristic=ga

deactivate