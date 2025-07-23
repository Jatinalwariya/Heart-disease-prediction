import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models'))

import LR 
import knn
import Decisiontree
import randomforest

print("\n🔷 All models have been run and evaluated successfully.")