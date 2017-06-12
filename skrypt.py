import os
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sciezka_katalogu = "C:/Users/Karina/dane/"


def wczytaj_pliki(sciezka):
    for scr, nazwy_katlagow, nazwy_plikow in os.walk(sciezka):
        for plik in nazwy_plikow:
            sciezka_pliku = os.path.join(scr, plik)
            if os.path.isfile(sciezka_pliku):
                with open(sciezka_pliku, "r", encoding="latin-1") as file:
                    text = file.read()
                yield sciezka_pliku, text


def listdir(path):
    # https://stackoverflow.com/questions/16953842/using-os-walk-to-recursively-traverse-directories-in-python
    dirs = []
    for name in os.listdir(path):
        path_name = os.path.join(path, name)
        if os.path.isdir(path_name):
            dirs.append(name)
    return dirs


def main():
    print("Wczytywanie Danych")
    dane = DataFrame({'text': [], 'klasa': []})
    paths = os.path.abspath(sciezka_katalogu)
    sciezki = listdir(paths)[:2]

    for sciezka in sciezki:
        klasa = sciezka
        sciezka_folderu = os.path.join(sciezka_katalogu, sciezka)
        wiersze = []
        index = []
        for nazwa_pliku, text in wczytaj_pliki(sciezka_folderu):
            wiersze.append({'text': text, 'klasa': klasa})
            index.append(nazwa_pliku)
        dane = dane.append(DataFrame(wiersze, index=index))
    print("Suma danych:", len(dane))
    print("Trenownie algorytmu")
    dane_uczace, dane_testowe = train_test_split(dane, test_size=.2, random_state=0)
    print(len(dane_uczace), len(dane_testowe))
    vectorizer = CountVectorizer()
    nb = MultinomialNB()

    zliczenia = vectorizer.fit_transform(dane_uczace['text'].values)
    targets = dane_uczace['klasa'].values
    nb.fit(zliczenia, targets)

    print("Walidacja")
    zliczenia = vectorizer.transform(dane_testowe['text'].values)
    predykcje = nb.predict(zliczenia)

    print('Maciez pomyłek:')
    print(confusion_matrix(dane_testowe['klasa'].values, predykcje))


if __name__ == "__main__":
    main()




