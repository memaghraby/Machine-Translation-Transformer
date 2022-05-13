
from IBM_Model1.IBM_Model1 import IBM


ibm = IBM()
ibm.load()

ibm.test()
# for english_word in ibm.getEnglishDict().keys():
#         for spanish_word in  ibm.getSpanishDict().keys():
#             print(ibm.translate(english_word, spanish_word))