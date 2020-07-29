import os
import pickle
import re
import random
import requests
from selenium import webdriver

browser = webdriver.Chrome()

pcl_file = "images/all_data.pcl"
if os.path.exists(pcl_file):
    with open(pcl_file, "rb") as f:
        all_data = pickle.load(f)
else:
    all_data = dict()
print(len(all_data))
current_page = random.choice(range(1, 400000))


def download_image(src, savepath):
    try:
        img = requests.get(src)
        with open(savepath, "wb") as f:
            f.write(img.content)
    except:
        pass


def save_post_data(thread_id, post_id, text, thumbnail, full_image):
    post_data = {
        "text": text,
        "thumbnail": thumbnail,
        "full_image": full_image,
    }
    all_data[thread_id][post_id] = post_data
    try:
        download_image(full_image, f"images/{post_id}.{full_extension}")
    except:
        pass

    try:
        download_image(thumbnail, f"images/{post_id}.{thumbnail_extension}")
    except:
        pass


def save_thread_post(thread_id):
    thread_image = browser.find_element_by_xpath(
        "//div[@class='thread_image_box']"
    ).find_element_by_xpath(".//a")
    thread_img_src = thread_image.get_attribute("href")

    thread_thumbnail = thread_image.find_element_by_xpath(
        ".//img"
    ).get_attribute("src")
    thread_text = browser.find_element_by_xpath("//div[@class='text']").text
    save_post_data(
        thread_id, thread_id, thread_text, thread_thumbnail, thread_img_src
    )


def parse_thread(thread_link):
    print(f"{current_page}        {len(all_data)}", end="\r")
    thread_id = thread_link.split("/")[-2]
    if thread_id not in all_data:
        all_data[thread_id] = dict()
    else:
        return
    browser.get(thread_link)
    save_thread_post(thread_id)
    posts = browser.find_elements_by_xpath("//div[@class='post_wrapper']")
    for post in posts:
        post_id_box = post.find_element_by_xpath(
            ".//a[@data-function='quote']"
        )
        post_id = post_id_box.text
        if post_id in all_data[thread_id]:
            continue
        images = post.find_elements_by_xpath(
            ".//a[@class='thread_image_link']"
        )
        if not images:
            continue
        image = images[0]
        full_image = image.get_attribute("href")
        full_extension = full_image.split(".")[-1]
        thumbnail = image.find_element_by_xpath(".//img").get_attribute("src")
        thumbnail_extension = thumbnail.split(".")[-1]
        text = post.find_element_by_xpath(".//div[@class='text']").text
        # no real text
        if re.match("^>>\d+$", text) or len(text) < 15:
            continue

        link = post_id_box.get_attribute("href")
        save_post_data(thread_id, post_id, text, thumbnail, full_image)


while current_page < 400000:
    browser.get(f"https://archived.moe/pol/page/{current_page}")
    current_page += 1
    threads = browser.find_elements_by_xpath(
        '//a[contains(text(), "View") ' 'and @class=\'btnr parent\']'
    )
    thread_links = [t.get_attribute("href") for t in threads]

    thread_links = [t for t in thread_links if "thread" in t and "#" not in t]
    for thread_link in thread_links:
        try:
            parse_thread(thread_link)
        except Exception as ex:
            print(ex)
    with open("images/all_data.pcl", "wb") as f:
        pickle.dump(all_data, f)
