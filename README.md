# ALHE-projekt

## Temat 12. Naucz robota chodzić
Korzystając ze środowiska OpenAI Gym: https://gym.openai.com/envs/#box2d zaimplementuj poznany na wykładzie algorytm (np. ewolucyjny/genetyczny) by nauczyć dwunożnego robota chodzić. Zadanie wymaga zaznajomienia z ideą sieci neuronowych, których parametry należy optymalizować w ramach projektu. Uwaga: Proszę nie stosować metod uczenia ze wzmocnieniem, a jedynie algorytmy oparte o populacje / algorytmy przeszukiwania.

**Dokumentacja wstępna:** https://docs.google.com/document/d/1SGi361dUx475NzdJJFqGgjtpwL87CuuYX02KqkK1vzs/edit# 

## Kroki
1. Sklonuj pliki projektu używając komendy:
```
git clone https://github.com/kklipski/ALHE-projekt.git
```
2. Utwórz nowy projekt w JetBrains PyCharm (IDE używane przez autorów).
3. Dodaj do nowoutworzonego projektu pliki źródłowe ze sklonowanego repozytorium (pliki znajdujące się w folderze [src](src)).
4. Zainstaluj w swoim projekcie pakiety wymienione w pliku [requirements.txt](requirements.txt).
5. Jeśli występują problemy z instalacją pakietu *torch*, skorzystaj z poniższego sposobu:

   W terminalu w PyCharm (warunek: otwarty projekt) użyj (pierwszej) komendy wygenerowanej na stronie: https://pytorch.org/get-started/locally/
	
   Przykład dla konfiguracji: PyTorch Build: *Stable (1.1)*, Your OS: *Windows*, Package: *Pip*, Language: *Python 3.7*, CUDA: *10.0*:
```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
``` z
6. W celu przeprowadzenia uczenia ze wzmocnieniem należy uruchomić skrypt [main.py](src/ddpg/main.py) z folderu [ddpg](src/ddpg).
7. W celu przeprowadzenia uczenia ze wzmocnieniem połączonego z algorytmem ewolucyjnym należy uruchomić skrypt [test_evo.py](src/ddpg_evo/test_evo.py) z folderu [ddpg_evo}(src/ddpg_evo).

## Autorzy
- **Lipski Kamil** - [kklipski](https://github.com/kklipski)
- **Rzepka Karol** - [krzepka](https://github.com/krzepka)

## Źródła
- https://arxiv.org/pdf/1711.09846.pdf - Population Based Training of Neural Networks (article)
- https://deepmind.com/blog/population-based-training-neural-networks/ - Population based training of neural networks (blog post)
- https://github.com/vy007vikas/PyTorch-ActorCriticRL - PyTorch implementation of DDPG algorithm for continuous action reinforcement learning problem (used in the project)