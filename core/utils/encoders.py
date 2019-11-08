import numpy as np


class MyLabelEncoder(object):

    def __init__(self):

        self.nom_to_cat = {}
        self.cat_to_nom = {}

        self.base_type = None

    def fit(self, y_nom):

        self.nom_to_cat = {}
        self.cat_to_nom = {}
        cat = 0

        self.base_type = type(y_nom[0])

        max_n = len(list(set(y_nom)))

        for nom in y_nom:
            if nom not in self.nom_to_cat:
                self.nom_to_cat[nom] = cat
                self.cat_to_nom[cat] = nom
                cat += 1

                if cat == max_n:
                    break

    def transform(self, y_nom):

        y_cat = [self.nom_to_cat[nom] for nom in y_nom]
        return np.asarray(y_cat, dtype=int)

    def inverse_transform(self, y_cat):

        y_nom = [self.cat_to_nom[cat] for cat in y_cat]
        return np.asarray(y_nom, dtype=self.base_type)

    def fit_transform(self, y_nom):

        self.fit(y_nom)
        return self.transform(y_nom)
