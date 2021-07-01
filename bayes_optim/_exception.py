class AskEmptyError(Exception):
    """Exception raised if BO.ask yields an empty outcome

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message=(
            "Ask yields empty solutions. This could be caused when "
            "when constraints are too restrict or the search space is already enumerated."
        ),
    ):
        self.message = message
        super().__init__(self.message)


class FlatFitnessError(Exception):
    """Exception raised if a flat fitness landscape is observed

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message=(
            "Too many flat objective values observed. The optimization process is terminated."
        ),
    ):
        self.message = message
        super().__init__(self.message)


class RecommendationUnavailableError(Exception):
    """Exception raised if the the recommendation is not available

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, details: str = ""):
        self.message = f"Optimizer's recommendation is not yet available due to {details}."
        super().__init__(self.message)
