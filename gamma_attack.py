import os

import magic
import numpy as np
from secml.array import CArray

from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
import lief


from secml_malware.attack.blackbox.c_gamma_sections_evasion import CGammaSectionsEvasionProblem
from secml_malware.attack.blackbox.c_wrapper_phi import CWrapperPhi
from abc import abstractmethod
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from secml_malware.models import CClassifierEnd2EndMalware, CClassifierEmber
from secml_malware.models.c_classifier_sorel_net import CClassifierSorel

import ember
import requests
import numpy as np

class Classifier(CClassifier):
    def __init__(self):
        pass
    def _fit():
        pass
    def _forward():
        pass

# For simplicity, encapsulate all classifiers into APIs so that the attacking threads can obtain the classification results directly through the APIs.
class OnlineWrapperPhi(CWrapperPhi):
    def __init__(self,idx):
        classifier =  Classifier()
        self.extractor = ember.PEFeatureExtractor(2, print_feature_warning=False)
        self.idx = idx
        self.classifier =  classifier
        #super().__init__()

    def extract_features(self, x: CArray) -> CArray:
        x = x.atleast_2d()
        size = x.shape[0]
        features = []
        for i in range(size):
            x_i = x[i,:]
            length = x_i.find(x_i == 256)
            if length:
                x_i = x_i[0, :length[0]]
            x_bytes = bytes(x_i.astype(np.uint8).tolist()[0])
            features.append(np.array(self.extractor.feature_vector(x_bytes), dtype=np.float32))
        features = CArray(features)
        return features
    
    def predict(self, x: CArray, return_decision_function: bool = True):
        x = self.extract_features(x)
        r0 = []
        for i in range(x.shape[0]):
            arr = x[i,:].tondarray().tobytes().hex()
            # Turn Numpy as Json 
            data = {
                "X": arr,
                "idx": self.idx
            }
            # Send to Target API
            response = requests.post('http://127.0.0.1:5001/predict', json=data)
            r0.append([1-response.json()[0][0],response.json()[0][0]])
        scores = CArray(r0)
        labels = (scores > 0.5).astype(int)
        label = labels.argmax(axis=1).ravel()
        return (label, scores) if return_decision_function is True else label


if __name__ == "__main__":
    #Here to load the pre-extracted benign sections; you can use sections provided by https://github.com/bitsecurerlab/MAB-malware (in the docker)
    section_population = []
    for c in os.listdir("benign_section_content/"):
        with open("benign_section_content/"+c,'rb') as f:
            content = [i for i in f.read()]
        section_population.append(content)

    results = []
    folder = 'datasets/malware/'  #INSERT MALWARE IN THAT FOLDER
    #Different index indicates different target classifiers
    for index in range(0,8):
        net = OnlineWrapperPhi(index)
        attack = CGammaSectionsEvasionProblem(section_population[:100], net, population_size=10, penalty_regularizer=1e-6, iterations=10, hard_label=False, threshold=0.5)
        X = []
        y = []
        file_names = []
        count = 0
        for i, f in enumerate(os.listdir(folder)):
            path = os.path.join(folder, f)
            if "PE32" not in magic.from_file(path):
                continue
            with open(path, "rb") as file_handle:
                code = file_handle.read()
            x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
            _, confidence = net.predict(x, True)
    
            if confidence[0, 1].item() < 0.5:
                continue
            print(f"> Added {f} with confidence {confidence[0,1].item()}")
            X.append(x)
            conf = confidence[1][0].item()
            y.append([1 - conf, conf])
            file_names.append(path)
            count += 1
            if count>=160:
                break   
        print(f'Start: model {index}, samples {count}')
        engine = CGeneticAlgorithm(attack)
        r = []
        adv_examples = []
        count = 0
        for sample, label in zip(X, y):
            count += 1
            y_pred, adv_score, adv_ds, f_obj = engine.run(sample, CArray(label[1]))
            print(count,engine.confidences_)
            r.append(engine.confidences_)
        results.append(r)
        joblib.dump(results,constants.SAVE_FILES_DIR+f'gamma_result_{index}.pkl')
