class InvalidOptionError(Exception):
    def __init__(self, option, valid_options):
        self.option = option
        self.valid_options = valid_options

    def __str__(self):
        return (
            f"Invalid option '{self.option}' in configuration."
            + "\n\n Note: \n"
            + "Valid options are: \n- "
            + "\n- ".join(self.valid_options)
        )
