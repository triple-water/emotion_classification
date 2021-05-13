import sys, os, functools
from inspect import getframeinfo, stack


def log_record(logger_obj, _func=None):
    def log_decorator_info(func):
        @functools.wraps(func)
        def log_decorator_wrapper(*args, **kwargs):
            args_passed_in_function = [repr(a) for a in args]
            kwargs_passed_in_function = ["{}={}".format(k, v) for k, v in kwargs.items()]
            formatted_arguments = ", ".join(args_passed_in_function + kwargs_passed_in_function)
            py_file_caller = getframeinfo(stack()[1][0])
            extra_args = {'func_name_override': func.__name__,
                          'file_name_override': os.path.basename(py_file_caller.filename)}
            logger_obj.info("Arguments: {} - Begin function {}".format(formatted_arguments, func.__name__))
            try:
                value = func(*args, **kwargs)
                logger_obj.info("Returned: - End function {}!".format(func.__name__))
            except:
                logger_obj.error("Exception: {}".format(str(sys.exc_info()[1])))
                raise
            return value

        return log_decorator_wrapper

    if _func is None:
        return log_decorator_info
    else:
        return log_decorator_info(_func)
