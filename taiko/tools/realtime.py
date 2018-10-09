from collections import deque

__all__ = ['AnalogData']


class AnalogData(object):
    def __init__(self, max_len):
        self.__ax = deque([0.0] * max_len)
        self.__ay = deque([0.0] * max_len)
        self.__az = deque([0.0] * max_len)
        self.__gx = deque([0.0] * max_len)
        self.__gy = deque([0.0] * max_len)
        self.__gz = deque([0.0] * max_len)
        self.__max_len = max_len

    def add(self, data):
        self.__add_to_buf(self.__ax, data[0])
        self.__add_to_buf(self.__ay, data[1])
        self.__add_to_buf(self.__az, data[2])
        self.__add_to_buf(self.__gx, data[3])
        self.__add_to_buf(self.__gy, data[4])
        self.__add_to_buf(self.__gz, data[5])

    def __add_to_buf(self, buf, val):
        """
        Add "val" to the newest position of deque "buf". If overflow, pop out the oldest position one.
        :param buf: the one-axis buffer
        :param val: original new value
        :return:
        """
        if len(buf) < self.__max_len:
            buf.append(val)
        else:
            buf.popleft()
            buf.append(val)

    @property
    def window(self):
        return [list(self.__ax), list(self.__ay), list(self.__az),
                list(self.__gx), list(self.__gy), list(self.__gz)]