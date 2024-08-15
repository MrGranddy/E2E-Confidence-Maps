from sklearn.ensemble import RandomForestClassifier


class Segmentor:

    def __init__(self, n_estimators, seed=None):

        self.rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5, random_state=seed
        )

    def train(self, patches, labels):
        self.rf.fit(patches, labels)

    def predict(self, patches):
        return self.rf.predict(patches)
