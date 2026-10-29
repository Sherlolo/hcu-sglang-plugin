from contextlib import nullcontext

class ComputeTransport:
    """One prepared operand with declared rows and output completion policy.

    ``prepare_output`` exposes the compute result's outstanding work.
    The boundary completes it before ``return_output`` moves the result back;
    a move that performs the sum declares ``sum_in_return`` before compute.
    Residual update and consumer normalization stay with the boundary.
    """

    @staticmethod
    def scope():
        return nullcontext()

    is_overlapped = False
    defers_sum = False
    sum_in_return = False

    def release(self):
        pass

    def prepare_output(self, output):
        return output

    def return_output(self, output):
        return output
