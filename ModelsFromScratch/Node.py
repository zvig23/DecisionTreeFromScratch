class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index: str = feature_index
        self.threshold: float = threshold
        self.left: Node = left
        self.right: Node = right
        self.info_gain: float = info_gain

        # for leaf node
        self.value: float = value
