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

## CONTROLS ##

#python python_src/ResNetMain.py --optimizer=adam --id=CRD1 --dataset=cbis --preloaded_weights=False --epochs=100
#python python_src/ResNetMain.py --optimizer=adam --id=CRD1-ImageNet --dataset=cbis --preloaded_weights=True --epochs=100
#
#python python_src/XceptionNetMain.py --optimizer=adam --id=CXD1 --dataset=cbis --preloaded_weights=False --epochs=100
#python python_src/XceptionNetMain.py --optimizer=adam --id=CXD1-ImageNet --dataset=cbis --preloaded_weights=True --epochs=100
#
#python python_src/UNetSegMain.py --id=SUD1 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=100
#python python_src/UNetSegMain.py --id=SUD1-ImageNet --dataset=cbis_seg --optimizer=sgd --preloaded_weights=True --epochs=100

#python python_src/ResNetMain.py --optimizer=adam --id=CRF1-ImageNet --dataset=ddsm --preloaded_weights=True --epochs=100
#python python_src/XceptionNetMain.py --optimizer=adam --id=CXF1-ImageNet --dataset=ddsm --preloaded_weights=True --epochs=100

## Metaheuristics ##

#python python_src/NewResNetMain.py --id=CRD2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga
#python python_src/NewXceptionNetMain.py --id=CXD2-GA --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=ga

#python python_src/NewResNetMain.py --id=CRD2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso
#python python_src/NewXceptionNetMain.py --id=CXD2-PSO --optimizer=adam --preloaded_weights=True --epochs=100 --meta_heuristic=pso

#python python_src/NewResNetMain.py --id=CRF2-GA-2 --optimizer=adam --preloaded_experiment=CRF2-GA --epochs=100 --meta_heuristic=ga --dataset=ddsm
#python python_src/NewResNetMain.py --id=CRF2-PSO-2 --optimizer=adam --preloaded_experiment=CRF2-PSO --epochs=100 --meta_heuristic=pso --dataset=ddsm

#
#python python_src/NewUNetSegMain.py --id=SUD2-I-GA-2 --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --preloaded_experiment=MSUD001-ImageNet --epochs=100 --meta_heuristic=ga
#python python_src/NewUNetSegMain.py --id=SUD2-I-PSO-2 --optimizer=sgd --dataset=cbis_seg --preloaded_weights=True --preloaded_experiment=MSUD001-ImageNet --epochs=100 --meta_heuristic=pso
pip install segmentation_models

python python_src/UNetSegMain.py --id=SUD1 --dataset=cbis_seg --optimizer=sgd --preloaded_weights=False --epochs=100

python python_src/NewUNetSegMain.py --id=SUD2-GA-2 --optimizer=sgd --dataset=cbis_seg --preloaded_experiment=SUD1 --epochs=100 --meta_heuristic=ga
python python_src/NewUNetSegMain.py --id=SUD2-PSO-2 --optimizer=sgd --dataset=cbis_seg --preloaded_experiment=SUD1 --epochs=100 --meta_heuristic=pso

python python_src/NewXceptionNetMain.py --id=CXF2-PSO-2 --optimizer=adam --preloaded_experiment=CXF2-PSO --epochs=100 --meta_heuristic=pso --dataset=ddsm
python python_src/NewXceptionNetMain.py --id=CXF2-GA-2 --optimizer=adam --preloaded_experiment=CXF2-PSO --epochs=100 --meta_heuristic=ga --dataset=ddsm

