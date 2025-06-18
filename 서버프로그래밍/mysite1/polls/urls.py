from django.urls import path

from . import views

app_name = "polls" #-- 어플리케이션에도 별명 붙이기 가능.
urlpatterns = [
    # ex: /polls/
    path("", views.index, name="index"),
    # ex: /polls/5/
    path("<int:question_id>/", views.detail, name="detail"), #-- alias: 별명, -->url 바뀌더라도 이름으로 접근가능
    # ex: /polls/5/results/
    path("<int:question_id>/results/", views.results, name="results"),
    # ex: /polls/5/vote/
    path("<int:question_id>/vote/", views.vote, name="vote"),
]
from django.urls import path


