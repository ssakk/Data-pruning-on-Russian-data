# Data-prunning-on-Russian-data
## Предыдущие работы
До сих пор усилия по отсеиванию данных для обучения LLMs проводились вручную, основываясь на списке фильтров. При качественной обрезке данных можно сократить затрачиваемые на обучение ресурсы, поэтому данный вопрос так важен. В статье [When Less is More](https://arxiv.org/pdf/2309.04564.pdf) проводится сравнение трёх методов обрезки данных (data pruning methods): perplexity, Error L2-Norm и Memorization factor. Проводится обучение авторерегрессивной decoder-only модели с GPT архитектурой на датасете CommonCrawl 2022-го года. Модель обучают снова, обрезая данные описанными методами, беря 10%, 30%, 50% и 70% интервалы с начала и с конца выбора.  Для понимания эффективности метрик данные также обрезаются рандомно.Затем тот же процесс проводится для моделей, предобученных на определенную задачу (SST, MRPC, QQP, QNLI, RTE & WNLI). В результате, после огромного количества экспериментов, выясняется, что лучше всех себя проявляет самая простая метрика - perplexity. 
## Введение
Perplexity –  это мера отклонения сгенерированного текста от настоящего.
$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum \log_2(p(w_i))\right)$
В контексте языковых моделей перплексия оценивает, насколько хорошо модель предсказывает текст, и часто используется для оценки качества моделей в задачах NLP.
Перплексия определяется как экспоненцированное среднее значение отрицательного логарифма вероятности прогнозируемых слов, где логарифм берется по основанию 2. Меньшее значение перплексии указывает на то, что модель лучше предсказывает выборку, т.е. модель более уверена в своих предсказаниях. На практике это означает, что модель с низкой перплексией более точно предсказывает следующие слова в тексте.
## Цель исследования
В данной работе мы хотим проверить эффективность метрики perplexity на данных русского языка, обучая модель, обученную для задачи анализа тональности и авторегрессионную модель для генерации текста.
## Методы
* Обучение берта на задачу сентимент анализа: [модель](https://huggingface.co/cointegrated/rubert-tiny), [датасет](https://www.kaggle.com/competitions/sentiment-analysis-in-russian/overview)  
Файнтюним предобученную модель для задачи классификации, затем пруним данные 9-ю способами: берём 40%,70% и 90% снизу, затем их же сверху; также пруним рандомно: убираем 40%, 70% и 90% датасета. Всего ообучаем 10 моделей;
* Обучение gpt: [модель](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2), [датасет](https://github.com/TatianaShavrina/taiga_site/blob/master/corpus/nplus1.md)  
Файнтюним предобученную на русском gpt2 на скачанных данных новостных сми, затем фильтруем 2-мя способами: считая перплексию и рандомно, на 7 интервалах, отсекаем только верхнюю часть датасета без деления на bottom/top.
## Результаты
Оценка модели, обученной для задачи SST, по 4 основным метрикам:
![alt-текст](https://github.com/ssakk/Data-prunning-on-Russian-data/blob/main/bert_precision.png {width=45%})
![alt-текст](https://github.com/ssakk/Data-prunning-on-Russian-data/blob/main/bert_recall.png {width=45%})
![alt-текст](https://github.com/ssakk/Data-prunning-on-Russian-data/blob/main/bert_f1.png)
![alt-текст](https://github.com/ssakk/Data-prunning-on-Russian-data/blob/main/bert_accuracy.png)
Обучая изначально на маленьком датасете из 150-ти файлов ошибка была около 8. После первого обучения gpt на полном датасете перплексия на валидационной выборке стала 2.6, после урезания 30% выборки стала 3.5. Это не говорит в пользу теории, однако для русского языка это нормальный результат, так как в русском гораздо больше морфологии, чем в английском. Далее на графике представлены результаты очистки по перплексии и случайным образом от 10% до 90%:
![alt-текст](https://github.com/ssakk/Data-prunning-on-Russian-data/blob/main/rugpt_results.png )
