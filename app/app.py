
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Загружаю модель
model = load_model('s_model/model_mn.keras')

# Передаю список признаков, на которых обучена модель
FEATURES_FOR_RATIO = [
    'Плотность, кг/м3',
    'модуль упругости, ГПа',
    'Количество отвердителя, м.%',
    'Содержание эпоксидных групп,%_2',
    'Температура вспышки, С_2',
    'Поверхностная плотность, г/м2',
    'Модуль упругости при растяжении, ГПа',
    'Прочность при растяжении, МПа',
    'Потребление смолы, г/м2',
    'Угол нашивки',
    'Шаг нашивки',
    'Плотность нашивки'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    # Сохраняю введённые значения
    form_data = {}
    for feature in FEATURES_FOR_RATIO:
        form_data[feature] = request.form.get(feature, '')

    if request.method == 'POST':
        try:
            # Собираю только нужные 10 значений в правильном порядке
            input_data = []
            for feature in FEATURES_FOR_RATIO:
                value = float(request.form[feature])
                input_data.append(value)

            # Прогноз
            input_array = np.array([input_data])  # shape (1, 10)
            prediction = model.predict(input_array)[0][0]
            result = round(prediction, 4)

        except Exception as e:
            result = f"Ошибка: {str(e)}"

    return render_template(
        'index.html',
        features_for_ratio=FEATURES_FOR_RATIO,
        form_data=form_data,
        result=result
    )

if __name__ == '__main__':
    app.run(debug=True)
