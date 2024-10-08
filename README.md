# Sentiment analysis of reviews
Данный репозиторий является финальным проектом для курса MLOPs на платформе OTUS.

## Постановка задачи
Цель проекта автоматизировать оценку отзывов о ресторанах, полученных в ходе сбора данных с основных онлайн-платформ. 
Автоматическая оценка отзывов позволит своевременно информировать руководство и администрацию ресторанов об ухудшении сервиса, и тем самым своевременно принять меры по улучшению качества обслуживания. Это позволит снизить репутационные потери. 

Для того, чтобы система имела бизнес эффект от внедрения, точность оценки отзывов должна быть более 90%. 
Не предполагается большой нагрузки на систему, однако система должна быть устойчива и работоспособна при поступлении до 3х отзывов в минуту. 

## Данные
Данные для системы могут быть взяты из открытых источников, а также из других источников
https://www.kaggle.com/code/mannarmohamedsayed/european-restaurant-reviews

Предполагается, что данные для обучения, валидации и тестирования, а также обученные модели хранятся в s3 в Yandex Cloud
s3://bucket-mlops-project/

## Трекинг экспериментов и моделей
Трекинг экспериментов и моделей осуществлятся с помощью MLFlow. 
Данный сервис развернут на отдельной удаленной машине в Yandex Cloud. MLFlow хранит даные обо всех моделях, их метриках, полученных на hold-out выборке и на кросс-валидации. Исходя из этих метрик выбирается лучшая модель и подгружается в образ, развернутый в k8s. 

## Разворачивание системы
Для разворачивания обученных моделей использовался кластер k8s. 
Доступ к кластеру осуществлялся с помощью настроенного ingress сервиса, сбор метрик осуществлялся с помощью prometheus. 
Все сервисы разворачивались с помощью helm. 
