# Open the file in read mode
with open('scripts/artist_list.txt', 'r') as file:
  # Read each line in the file
  for line in file:
    # Split the line by commas
    parts = line.split(',')
    # If there are no commas, use the whole line
    text = parts[0] if len(parts) > 0 else line
    # Write the text back to the file
    with open('scripts/artist_list_fixup.txt', 'a') as output_file:
      output_file.write(text.strip() + '\n')