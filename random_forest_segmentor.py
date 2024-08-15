from sklearn.ensemble import RandomForestClassifier


class Segmentor:

    def __init__(self, seed=None):

        self.rf = RandomForestClassifier(
            n_estimators=25, max_depth=5, random_state=seed
        )

    def train(self, patches, labels):
        self.rf.fit(patches, labels)

    def predict(self, patches):
        return self.rf.predict(patches)
