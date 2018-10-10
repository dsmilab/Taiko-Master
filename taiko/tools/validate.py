__all__ = ['validate_integer',
           'validate_alpha_digit']


def validate_integer(value_if_allowed, ch='0'):
    if value_if_allowed == '':
        return True

    if ch.isdigit():
        try:
            val = int(value_if_allowed)
            if 0 <= val < 100:
                return True
        except ValueError:
            pass

    return False


def validate_alpha_digit(value_is_allowed, ch='0'):
    if len(value_is_allowed) > 12:
        return False

    if ch.isdigit() or ch.isalpha() or ch == '_':
        return True
    return False
