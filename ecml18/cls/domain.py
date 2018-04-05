
import pymzn
import numpy as np
from cls.utils import *


class Domain(object):
    """A domain encoded by MiniZinc attributes and constraints.

    A domain is a class that feeds the Models with the MiniZinc representation
    of the objects attributes, the constraints of the search space and the
    features of the feature map.

    This is the base Domain class. Users of this class can use it for static
    context-less domains.  Inherit this class to provide a meaningful context
    drawing function or to generate a parameter-dependent domain.  Default
    domains are available in the `weaver.domains` package.

    Arguments
    ---------
    template : str
        The template string or file name of the MiniZinc models.
        The template can be used as a preamble for each problem, including
        calculating phi, inference and query selection. It should contain things
        like context parameter declaration, function and predicate declaration,
        initial data wrangling.
    attributes : dict
        The MiniZinc attributes of the objects in a dictionary of the type
        {attr: attr_type}.
    constraints : list of str
        The MiniZinc constraints of the search space.
    features : list of str
        The MiniZinc features included in the feature map.
        This method should return a list of all available features. When using
        critiquing methods, users features will be sampled from this list.
    feat_type : str
        The MiniZinc type of the feature vector.
    """
    def __init__(self, template, attributes, constraints, features,
                 feat_type='float'):
        self.template = template
        self.attributes = attributes
        self.constraints = constraints
        self.features = features
        self.feat_type = feat_type
        self._phis = {}

    def phi(self, x, y, features=None):
        """The feature map.

        Returns the feature vector of the object y in the context x with respect
        to the given features. This function is cached, there is an overhead for
        computing the feature vector only the first time it is called with the
        same parameters.

        Parameters
        ----------
        x : dict
            The context of the current iteration. This dictionary is given to
            the MiniZinc problem as additional data.
        y : dict
            The object to calculate the feature vector on.
        features : list
            The list of features to calculate the feature vector with.

        Returns
        -------
        numpy.ndarray
            The computed feature vector.
        """
        if features is None:
            features = self.features

        _frx = (freeze(x), freeze(y), tuple(sorted(features)))
        if _frx in self._phis:
            return self._phis[_frx]
        print("called")
        model = pymzn.MiniZincModel(self.template)
        model.parameters(y.items())

        feat_type = self.feat_type
        model.parameter('FEATURES', mzn_range(features))
        model.array_variable('phi', 'FEATURES', feat_type, features, output=True)
        model.satisfy()
        sol = pymzn.minizinc(model, data=x)
        _phi = sol[0]['phi']
        self._phis[_frx] = np.array(_phi, dtype=np.float64)
        return self._phis[_frx]


    def infer(self, x, w, features=None):
        """Solve an inference problem.

        Returns the object y that maximizes the utility with respect to the
        given weights w and the given features, in context x.

        Parameters
        ----------
        x : dict
            The context of the current iteration. This dictionary is given to
            the MiniZinc problem as additional data.
        w : numpy.ndarray
            The weight vector to use for the utility function.
        features : list
            The list of features to calculate the feature vector with.

        Returns
        -------
        dict
            The optimal object y.
        """
        if features is None:
            features = self.features

        model = pymzn.MiniZincModel(self.template)

        # CONVERTIRE IN PESI INTERI if self.feat_type == 'int'
        model.parameter('w', w)
        for attr, attr_type in self.attributes.items():
            model.variable(attr, attr_type)

        feat_type = self.feat_type
        model.parameter('FEATURES', mzn_range(features))
        model.array_variable('phi', 'FEATURES', feat_type, features)

        model.constraints(self.constraints)

        dtype = dot_type(w, feat_type)
        model.variable('obj', dtype, mzn_dot('w', 'phi'))
        model.maximize('obj')
        return pymzn.minizinc(model, data=x, parse_output=False, parallel=8)[0]

    def draw_context(self, user, it):
        """Draw a context x for the given user.

        This method needs to be implemented only if the domain is used in an
        online setting.

        Parameters
        ----------
        user : User
            The user who draws the context.
        """
        return {}

    def sample_features(num_features, *args, **kwargs):
        raise NotImplementedError()

