#pylint: disable=no-member

import autodiff
import autodiff.numpy_grad.wrapper as wnp

if __name__ == "__main__":
    print(wnp.add(1,2))