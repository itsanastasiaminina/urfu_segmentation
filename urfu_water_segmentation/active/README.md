## Окружение
В папку urfu_water_segmantation/urfu_segmentation/urfu_water_segmentation
```sh
sh init_venv.sh
```
Далее для активации окружения можно использовать
```sh
source .venv/bin/activate
```
## Датасет 
Для формирование датасета используется файл data_genrator.py, в переменную base_path необходимо указать путь до датасета
```sh
python active/data_genrator.py
```
## Активное обучение 
Для старта активного обучения настройте параметры, выписаны капсом в файле active_loop и также найдите переменную DEFAULT_AL_BATCH
```sh
sbatch -n1 -p hiperf --nodelist=tesla-a101 --gres=gpu:1 --cpus-per-task=12 --mem=100000 -t 20:00:00 --job-name=active-learn --output=./logs/train"_%j" --wrap="TORCH_SHOW_CPP_STACKTRACES=1 python -X faulthandler  active/active_loop.py"
```
- часть конфига редачится в файлe active_loop
## Метрики
В файле active_loop 122-128 можно раскоммментировать строки и будут считаться метрики на всем датасете, затратная операция, требует гибкой работы с батчем
