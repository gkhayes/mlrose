""" Classes for defining fitness functions."""

# Author: Genevieve Hayes (Modified by Andrew Rollings)
# License: BSD 3 clause


class _DiscretePeaksBase:

    @staticmethod
    def head(_b, _x):
        """Determine the number of leading b's in vector x.

        Parameters
        ----------
        b: int
            Integer for counting at head of vector.
        x: array
            Vector of integers.

        Returns
        -------
        head: int
            Number of leading b's in x.
        """

        # Initialize counter
        _head = 0

        # Iterate through values in vector
        for i in _x:
            if i == _b:
                _head += 1
            else:
                break

        return _head

    @staticmethod
    def tail(_b, _x):
        """Determine the number of trailing b's in vector x.

        Parameters
        ----------
        b: int
            Integer for counting at tail of vector.

        x: array
            Vector of integers.

        Returns
        -------
        tail: int
            Number of trailing b's in x.
        """

        # Initialize counter
        _tail = 0

        # Iterate backwards through values in vector
        for i in range(len(_x)):
            if _x[len(_x) - i - 1] == _b:
                _tail += 1
            else:
                break

        return _tail

