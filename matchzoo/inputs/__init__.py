# note
import six
from keras.utils.generic_utils import deserialize_keras_object

from point_generator import PointGenerator
from point_generator import Triletter_PointGenerator
from point_generator import DRMM_PointGenerator

from pair_generator import PairGenerator
from pair_generator import Triletter_PairGenerator
from pair_generator import DRMM_PairGenerator
from pair_generator import PairGenerator_Feats
from pair_generator import DMN_PairGenerator
from pair_generator import DMN_KD_PairGenerator
from pair_generator import DMN_PairGeneratorWithIntents
from pair_generator import DMN_PairGeneratorOnlyIntents
from pair_generator import DMN_PairGenerator_Web_v2

from list_generator import ListGenerator
from list_generator import Triletter_ListGenerator
from list_generator import DRMM_ListGenerator
from list_generator import ListGenerator_Feats
from list_generator import DMN_ListGenerator
from list_generator import DMN_ListGeneratorWithIntents
from list_generator import DMN_ListGeneratorOnlyIntents
from list_generator import DMN_KD_ListGenerator
from list_generator import DMN_ListGenerator_Web_v2

def serialize(generator):
    return generator.__name__

def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='input function')

def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'input function identifier:', identifier)

