import logging

def build_logging(filename, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger

if __name__ == '__main__':
    logger1 = build_logging('log1.txt', 'Hyp1')
    printer1 = logger1.info
    printer1(1)

    logger2 = build_logging('log2.txt', 'Hyp2')
    printer2 = logger2.info
    printer2(2)
   

