class NICE_ClassNotFound(Exception):

    def __init__(self):

        self.message = (
            "String does not match a default Class."
            )

        super().__init__(self.message)

class NICE_NodeNotFound(Exception):

    def __init__(self, node = ''):

        self.message = (
            f"Node {node} not found. Nodes must be added before assets which belong to them."
            )

        super().__init__(self.message)

class NICE_InvalidBaseClass(Exception):

    def __init__(self):

        self.message = (
            "Invalid base class."
            )

        super().__init__(self.message)