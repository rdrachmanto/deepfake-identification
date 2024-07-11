# Append the root folder in sys.path, ensuring access to src/ in the notebooks
import sys
import logging
from datetime import datetime

sys.path.append('../')
logging.basicConfig(
    filename=f'./logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)