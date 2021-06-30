class AskEmptyError(Exception):
    """Exception raised if BO.ask yields an empty outcome

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message=(
            "Ask yields empty solutions. Please check the search space/constraints are feasible"
        ),
    ):
        self.message = message
        super().__init__(self.message)
