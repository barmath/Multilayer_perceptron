import csv

## TODO
## add a method to convert 0 into 1

class Csv_maneger:
	"""Read a CSV file and store it in to a array.

    This class open store and read a CSV file return a array of all the lines.
    """
	def read_csv(name_of_file):
		"""Reads, opens and format the CSV file.
    
        It basically check if the name is not empty ;
		Then open and traverse to all teh lines of the CSV file ;
		and stores it in a array.

        Args:
            String with the name of the file to open.

        Returns:
            A list of list of integers with all the data in the requested file.
        """
		if name_of_file is None or len(name_of_file) == 0:
			return []

		converted = []

		with open(name_of_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter = ',')
			for row in csv_reader:
				converted.append(row)

		return converted
