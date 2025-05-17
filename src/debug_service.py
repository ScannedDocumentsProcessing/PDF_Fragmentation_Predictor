from main import MyService

def main():
    service = MyService()

    with open("data/raw_data/jossef/10.pdf", "rb") as pdf_file:
        pdf_data = pdf_file.read()
        print(service.predict(pdf_data))


if __name__ == "__main__":
    main()