import sys
import csv
def main():
    sent_file = open(sys.argv[1])
    csv_file = open(sys.argv[2])
    file_reader = csv.reader(csv_file)
    #TODO: Implement

if __name__ == '__main__':
    main()
