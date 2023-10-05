echo cwd: $(pwd)
# Загрузка raw датасета, очистка
python src/make_dataset.py --stage_="all"
# Feature Engineering
python src/build_features.py --stage_="build_features_cnn"
# Обучение и оценка модели
python src/train_model.py --stage_="train_model_transform" --type_='cnn'
# Предикт типа вина
python src/gender/predict_model.py --type_=="cnn"