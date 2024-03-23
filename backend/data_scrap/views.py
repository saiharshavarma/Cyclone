import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
# from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import urllib

def fetchRealTimeData():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.google.com/")

    driver.get('https://mausam.imd.gov.in/imd_latest/contents/satellite.php')

    time.sleep(5)

    image_element = driver.find_element(By.XPATH, '//*[@id="images"]/img')
    src = image_element.get_attribute('src')
    try:
        print("Processing Image Success")
        urllib.request.urlretrieve(src, 'static/real-time-scrap/processing.png')
    except Exception as e:
        print("Done")

    driver.get("https://tropic.ssec.wisc.edu/real-time/imagemain.php?&basin=indian&prod=irbbm&sat=m5")

    time.sleep(5)

    image_element = driver.find_element(By.XPATH, '/html/body/table/tbody/tr/td/center/p/img')
    src = image_element.get_attribute('src')
    try:
        print("Original Image Success")
        urllib.request.urlretrieve(src, 'static/real-time-scrap/original.png')
    except Exception as e:
        print("Done")    

    driver.get("https://tropic.ssec.wisc.edu/real-time/imagemain.php?&basin=indian&prod=irn&sat=m5")

    time.sleep(5)

    image_element = driver.find_element(By.XPATH, '/html/body/table/tbody/tr/td/center/p/img')
    src = image_element.get_attribute('src')
    try:
        print("IR Image Success")
        urllib.request.urlretrieve(src, 'static/real-time-scrap/ir.png')
    except Exception as e:
        print("Done")
    
    driver.quit()
    print("Sleeping")