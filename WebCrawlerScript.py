from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchWindowException, TimeoutException, WebDriverException, \
    NoSuchElementException, StaleElementReferenceException
import time
import csv
import os
import glob
import pandas as pd
import chromedriver_autoinstaller

# Automatically download & use the correct driver version for your Chrome
driver_path = chromedriver_autoinstaller.install()
print("✅ ChromeDriver installed at:", driver_path)

# Set ChromeDriver path
# driver_path = "D:/Users/Yuqi‘s Rog/Work_Stu/chromedriver-win64/chromedriver.exe"
service = Service(driver_path)
# driver = webdriver.Chrome(service=service)

from selenium.webdriver.chrome.options import Options



download_count = 0  # Record the number of downloads
processed_links = set()  # Set of processed facility links

download_directory = "/Users/jacobfernandez/Desktop/Research/FA2025/downloads"
station_data = []

options = Options()
prefs = {"download.default_directory": download_directory,
         "download.prompt_for_download": False,
         "directory_upgrade": True,
         "safebrowsing.enabled": True}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(service=service, options=options)

# Get the latest CSV file
def get_latest_downloaded_file(directory):
    list_of_files = glob.glob(f"{directory}/*.csv")
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    return None

# Wait for download to complete
def wait_for_download_to_complete(directory, timeout=60):
    elapsed_time = 0
    while elapsed_time < timeout:
        latest_file = get_latest_downloaded_file(directory)
        if latest_file and not latest_file.endswith('.crdownload'):
            return latest_file
        time.sleep(1)
        elapsed_time += 1
    return None

# Apply filters on the main page (only applied at initial load)
def apply_filters():
    try:
        # Try different selectors to find the "solar" button
        try:
            solar_label = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
                (By.XPATH, "//label[@for='searchByTechnologySolar']")))
        except TimeoutException:
            solar_label = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "label[aria-label='Facilities with solar photovoltaic technology']")))
        solar_label.click()
        time.sleep(10)  # Add more waiting time to ensure page fully loads

        # Select "show all results"
        try:
            show_all_dropdown = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'facility-count-select')))
        except TimeoutException:
            show_all_dropdown = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "select#facility-count-select")))
        select = Select(show_all_dropdown)
        select.select_by_value('9999')
        time.sleep(20)  # Add more waiting time to ensure data fully loads
    except Exception as e:
        print(f"Error applying filters: {e}")

try:
    # 1. Open the main page
    driver.get('https://der.nyserda.ny.gov/search/')
    wait = WebDriverWait(driver, 10)  # Longer wait time

    # Initial filter application
    apply_filters()

    # Get starting index
    while True:
        try:
            detail_buttons = driver.find_elements(By.XPATH, "//img[@alt='Click here to view facility details']")
            print(f"Number of detail buttons found: {len(detail_buttons)}")
            start_index = int(input("Enter the starting index for downloading facility links: "))
            if 0 <= start_index < len(detail_buttons):
                break
            else:
                print("Index out of range, please enter a valid index.")
        except ValueError:
            print("Please enter a valid integer.")

    # Get facility detail buttons
    def get_detail_buttons():
        buttons = driver.find_elements(By.XPATH, "//img[@alt='Click here to view facility details']")
        print(f"Number of detail buttons re-fetched: {len(buttons)}")
        return buttons

    detail_buttons = get_detail_buttons()

    while start_index < len(detail_buttons):
        # Re-fetch detail buttons to avoid stale element errors
        detail_buttons = get_detail_buttons()
        for index in range(start_index, len(detail_buttons)):
            try:
                button = detail_buttons[index]

                # Scroll to button position and click
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", button)
                time.sleep(5)  # Wait for page to load

                # Get address info (update to extract full text from <div class="address">)
                try:
                    address_element = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.address"))
                    )
                    # Get child text and remove blank lines
                    address_lines = address_element.get_attribute("innerText").strip().split("\n")
                    address = ", ".join([line.strip() for line in address_lines if line.strip()])
                except NoSuchElementException:
                    address = ""
                    print("Failed to get address info, continue processing.")

                # Get "view data" links and process data
                view_data_links = driver.find_elements(By.XPATH, "//a[contains(@class, 'img-graph') and text()=' View Data']")
                if len(view_data_links) > 1:
                    driver.execute_script("arguments[0].scrollIntoView(true);", view_data_links[1])
                    time.sleep(1)
                    view_data_links[1].click()
                    time.sleep(5)

                    # Click "Data" tab and select "HR"
                    try:
                        data_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
                            (By.XPATH, "//li[@id='download-data-tab' and contains(text(), 'Data')]")))
                        data_button.click()
                        time.sleep(5)

                        hr_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
                            (By.XPATH, "//label[@for='download-data-control-tstep-hour']")))
                        hr_button.click()
                        time.sleep(2)

                    except (TimeoutException, NoSuchElementException):
                        print(f"Failed to find Data or HR button, skipping facility link {index}")
                        driver.back()
                        time.sleep(5)
                        continue

                    # Click "Download" button
                    try:
                        download_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
                            (By.XPATH, "//button[@title='Download all performance data from this project in CSV format']")))
                        driver.execute_script("arguments[0].click();", download_button)
                        time.sleep(10)

                        # Wait for CSV to finish downloading and write address info to it
                        csv_file_path = wait_for_download_to_complete(download_directory)
                        if csv_file_path:
                            # Ensure file is stable before writing
                            time.sleep(2)

                            try:
                                # Open CSV with pandas
                                df = pd.read_csv(csv_file_path)

                                # Ensure at least 3 rows exist
                                while len(df) <= 2:
                                    df.loc[len(df)] = [None] * max(8, len(df.columns))

                                # Add columns if fewer than 8
                                while len(df.columns) < 8:
                                    df[f'Unnamed_{len(df.columns)}'] = None

                                # Set cell H3 (row 3, column 8) to address
                                df.iat[2, 7] = address

                                # Save back to CSV
                                df.to_csv(csv_file_path, index=False)
                                print(f"Address info written to H3 cell of {csv_file_path}.")
                            except Exception as e:
                                print(f"Error writing address to H3 cell: {e}")

                            print(f"Downloaded facility link {index} and updated CSV file.")
                        else:
                            print("No downloaded CSV file found")

                    except (TimeoutException, NoSuchElementException):
                        print(f"Failed to find Download button, skipping facility link {index}")
                        driver.back()
                        time.sleep(5)
                        continue

                # Return to previous page
                retries = 3  # Set retry attempts
                while retries > 0:
                    try:
                        driver.back()
                        time.sleep(10)  # Extra wait to ensure page loads
                        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'facility-count-select')))
                        detail_buttons = get_detail_buttons()  # Re-fetch button list
                        start_index = index + 1
                        break
                    except (TimeoutException, StaleElementReferenceException):
                        retries -= 1
                        if retries == 0:
                            raise

            except (NoSuchWindowException, WebDriverException, NoSuchElementException, StaleElementReferenceException,
                    TimeoutException) as e:
                print(f"Error processing facility link {index}: {e}")
                driver.get('https://der.nyserda.ny.gov/search/')
                time.sleep(10)
                apply_filters()
                detail_buttons = get_detail_buttons()
                start_index = index + 1
                break

finally:
    driver.quit()
