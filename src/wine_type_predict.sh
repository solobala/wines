echo cwd: $(pwd)
# Загрузка raw датасета, очистка
python src/make_dataset.py --stage_=="all"
# Feature Engineering
python src/build_features.py --stage_="build_features"
# Обучение бинарной модели
python src/train_model.py --stage_="train_model" --type_='binary'
# Оценка бинарной модели
python src/train_model.py --stage_="eval_model" --type_='binary'
# Предикт типа вина
python src/gender/predict_model.py --type_=="binary"