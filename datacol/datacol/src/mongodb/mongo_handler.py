from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, DuplicateKeyError
from datetime import datetime
import logging


class MongoDBHandler:
    def __init__(self, db_name="predict_stock"):
        # Connection string for MongoDB Atlas
        self.client = MongoClient(
            "mongodb+srv://Lubabah:1234@cluster0.o5hydr7.mongodb.net/?retryWrites=true&w=majority&appName=cluster0")

        # Choose the database
        self.db = self.client[db_name]

        # We'll select collections dynamically based on data type
        self.collections = {}

        # Keep track of collections with TTL indexes
        self.ttl_collections = set()

        # Keep track of collections with unique indexes
        self.unique_collections = set()

    def get_collection(self, data_type):
        """
        Get or create a collection for the specified data type.
        """
        if data_type not in self.collections:
            self.collections[data_type] = self.db[data_type]
        return self.collections[data_type]

    def find_documents(self, collection_name, query, projection=None, limit=None, sort=None):
        """
        Find documents in a collection based on a query.

        Parameters:
        - collection_name: The name of the collection to query
        - query: The MongoDB query dictionary
        - projection: Optional dictionary of field projections
        - limit: Optional maximum number of documents to return
        - sort: Optional list of (field, direction) pairs for sorting

        Returns:
        - List of documents matching the query
        """
        try:
            collection = self.get_collection(collection_name)

            # Build the find cursor with the query and projection
            cursor = collection.find(query, projection)

            # Apply sort if provided
            if sort:
                cursor = cursor.sort(sort)

            # Apply limit if provided
            if limit:
                cursor = cursor.limit(limit)

            # Convert cursor to list of documents
            documents = list(cursor)

            logging.info(f"Found {len(documents)} documents in {collection_name} collection.")
            return documents

        except Exception as e:
            logging.error(f"Error finding documents in {collection_name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []

    def ensure_unique_index(self, collection_name, field_name='_id'):
        """
        Ensures a unique index exists on the specified field for a collection.
        """
        if collection_name in self.unique_collections:
            return

        try:
            collection = self.get_collection(collection_name)
            collection.create_index([(field_name, 1)], unique=True)
            logging.info(f"Created unique index on {field_name} for {collection_name}")
            self.unique_collections.add(collection_name)
        except Exception as e:
            logging.error(f"Error creating unique index for {collection_name}: {e}")

    def insert_data(self, data, data_type, apply_ttl=False, handle_duplicates=True):
        """
        Insert data into the appropriate MongoDB collection based on data type.
        Data must be a dict or a list of dicts.

        Parameters:
        - data: A dict or list of dicts to insert
        - data_type: The collection name to insert into
        - apply_ttl: If True, adds a created_at timestamp for TTL expiration
        - handle_duplicates: If True, will skip documents that would cause duplicate key errors
        """
        try:
            # Get the correct collection
            collection = self.get_collection(data_type)

            # Add detailed logging about the data
            logging.info(f"Inserting data into {data_type} collection")
            if data is None:
                logging.warning(f"Attempted to insert None data into {data_type} collection. Skipping.")
                return False

            # Add 'created_at' timestamp if TTL should be applied
            if apply_ttl:
                # Ensure the collection has a TTL index
                if data_type not in self.ttl_collections:
                    self.create_ttl_index(data_type)

                # Add timestamp to documents
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item['created_at'] = datetime.utcnow()
                elif isinstance(data, dict):
                    data['created_at'] = datetime.utcnow()

            # Handle list of items
            if isinstance(data, list):
                # Check if the list is empty
                if not data:
                    logging.warning(f"Empty list provided to insert_data for {data_type}. Nothing to insert.")
                    return False

                # Ensure each item in the list is a dictionary
                valid_docs = []
                for item in data:
                    if isinstance(item, dict):
                        valid_docs.append(item)
                    else:
                        try:
                            # Try converting to dict if it has attributes
                            dict_item = vars(item)
                            valid_docs.append(dict_item)
                        except (TypeError, AttributeError):
                            logging.error(f"Could not convert item to dict: {item}")

                if valid_docs:
                    # Insert in batches to avoid potential issues with large datasets
                    batch_size = 500
                    inserted_count = 0
                    duplicate_count = 0

                    for i in range(0, len(valid_docs), batch_size):
                        batch = valid_docs[i:i + batch_size]
                        try:
                            result = collection.insert_many(batch, ordered=False)
                            inserted_count += len(result.inserted_ids)
                        except BulkWriteError as bwe:
                            if handle_duplicates:
                                # Count how many were actually inserted despite the error
                                if hasattr(bwe, 'details') and 'nInserted' in bwe.details:
                                    inserted_count += bwe.details['nInserted']
                                    duplicate_count += len(batch) - bwe.details['nInserted']
                                logging.warning(f"Some documents were duplicates and were skipped: {bwe.details}")
                            else:
                                raise

                    logging.info(
                        f"Inserted {inserted_count} documents into {data_type} collection (skipped {duplicate_count} duplicates).")
                    return True
                else:
                    logging.error(
                        f"No valid documents to insert into {data_type} collection after conversion attempts.")
                    return False

            # Handle single item
            elif isinstance(data, dict):
                try:
                    result = collection.insert_one(data)
                    logging.info(f"Inserted 1 document into {data_type} collection with ID: {result.inserted_id}")
                    return True
                except DuplicateKeyError:
                    if handle_duplicates:
                        logging.warning(f"Document already exists in {data_type} collection, skipped.")
                        return False
                    else:
                        raise
            else:
                # Try to convert to dict if it has attributes
                try:
                    dict_data = vars(data)
                    try:
                        result = collection.insert_one(dict_data)
                        logging.info(
                            f"Inserted converted object into {data_type} collection with ID: {result.inserted_id}")
                        return True
                    except DuplicateKeyError:
                        if handle_duplicates:
                            logging.warning(f"Document already exists in {data_type} collection, skipped.")
                            return False
                        else:
                            raise
                except (TypeError, AttributeError):
                    logging.error(
                        f"Cannot insert data of type {type(data)} into {data_type} collection. Must be a dict or an object with dict attributes.")
                    raise TypeError("Document must be a dict, list of dicts, or convertible to dict")

        except Exception as e:
            logging.error(f"Failed to insert data into {data_type} MongoDB collection: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def upsert_data(self, data, data_type, id_field=None, apply_ttl=False):
        """
        Upsert data into MongoDB (update if exists, insert if not).
        This is useful for data types where you want to update existing records.

        Parameters:
        - data: A dict or list of dicts to upsert
        - data_type: The collection name to upsert into
        - id_field: Field to use as unique identifier (defaults to natural identifier)
        - apply_ttl: If True, adds a created_at timestamp for TTL expiration
        """
        try:
            collection = self.get_collection(data_type)

            logging.info(f"Upserting data into {data_type} collection")
            if data is None:
                logging.warning(f"Attempted to upsert None data into {data_type} collection. Skipping.")
                return False

            # Determine common ID fields based on data type if not specified
            if id_field is None:
                if data_type == 'stock':
                    id_field = 'symbol'  # Use stock symbol as ID
                elif data_type == 'news':
                    id_field = 'url'  # Use URL as ID
                elif data_type == 'economic':
                    id_field = 'indicator'  # Use indicator name as ID
                elif data_type == 'twitter':
                    id_field = 'tweet_id'  # Use tweet ID as ID
                elif data_type == 'reddit':
                    id_field = 'post_id'  # Use post ID as ID
                else:
                    id_field = '_id'  # Default to MongoDB's _id

            # Determine if we're dealing with a list or single item
            documents = data if isinstance(data, list) else [data]

            if not documents:
                logging.warning(f"Empty data provided for upsert in {data_type}. Nothing to upsert.")
                return False

            # Add 'created_at' timestamp if TTL should be applied and not already present
            if apply_ttl:
                # Ensure the collection has a TTL index
                if data_type not in self.ttl_collections:
                    self.create_ttl_index(data_type)

                # Add timestamp to documents if not already present
                now = datetime.utcnow()
                for doc in documents:
                    if 'created_at' not in doc:
                        doc['created_at'] = now

            # Add 'updated_at' timestamp to all documents
            now = datetime.utcnow()
            for doc in documents:
                doc['updated_at'] = now

            # Prepare bulk operations
            bulk_operations = []

            for doc in documents:
                # Skip documents without the required ID field
                if id_field not in doc or doc[id_field] is None:
                    continue

                # Create filter criteria
                filter_criteria = {id_field: doc[id_field]}

                # Use the $set operator to update only the fields in the document
                update_operation = {
                    '$set': doc
                }

                # Add to bulk operations
                bulk_operations.append(
                    UpdateOne(filter_criteria, update_operation, upsert=True)
                )

            # Execute bulk operations
            if bulk_operations:
                result = collection.bulk_write(bulk_operations)
                logging.info(f"Upsert results for {data_type}: "
                             f"modified={result.modified_count}, "
                             f"upserted={len(result.upserted_ids)}")
                return True
            else:
                logging.warning(f"No valid documents to upsert in {data_type} collection")
                return False

        except Exception as e:
            logging.error(f"Failed to upsert data into {data_type} MongoDB collection: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def create_ttl_index(self, collection_name):
        """
        Creates a TTL index on the 'created_at' field to automatically delete documents after 24 hours.

        This will only affect documents that have a 'created_at' field.
        Documents without this field will not be affected by the TTL index.
        """
        try:
            collection = self.get_collection(collection_name)

            # Check if TTL index already exists
            existing_indexes = collection.index_information()
            ttl_index_exists = False

            for idx_name, idx_info in existing_indexes.items():
                if 'expireAfterSeconds' in idx_info and idx_info.get('key') == [('created_at', 1)]:
                    ttl_index_exists = True
                    break

            # Create TTL index if it doesn't exist
            if not ttl_index_exists:
                collection.create_index([("created_at", 1)], expireAfterSeconds=86400)  # 86400 seconds = 24 hours
                logging.info(f"TTL index created on 'created_at' field for {collection_name} collection.")
                self.ttl_collections.add(collection_name)
            else:
                logging.info(f"TTL index already exists for {collection_name} collection.")
                self.ttl_collections.add(collection_name)

            return True
        except Exception as e:
            logging.error(f"Failed to create TTL index for {collection_name}: {e}")
            return False

    def mark_document_for_expiration(self, collection_name, document_id, ttl_hours=24):
        """
        Marks a specific document for TTL expiration by adding a created_at timestamp.

        Parameters:
        - collection_name: Name of the collection
        - document_id: The MongoDB _id of the document
        - ttl_hours: How many hours before document should expire (default 24)
        """
        try:
            # Ensure the collection has a TTL index
            if collection_name not in self.ttl_collections:
                self.create_ttl_index(collection_name)

            # Update the document to add the created_at field
            collection = self.get_collection(collection_name)
            result = collection.update_one(
                {'_id': document_id},
                {'$set': {'created_at': datetime.utcnow()}}
            )

            if result.modified_count > 0:
                logging.info(
                    f"Document {document_id} in {collection_name} marked for expiration after {ttl_hours} hours.")
                return True
            else:
                logging.warning(f"Document {document_id} not found in {collection_name} or already has expiration.")
                return False

        except Exception as e:
            logging.error(f"Failed to mark document for expiration: {e}")
            return False

    def remove_duplicates(self, collection_name, key_field='url'):
        """
        Removes duplicate documents from a collection based on a key field.
        Keeps the newest document for each duplicate set.

        Parameters:
        - collection_name: The name of the collection to process
        - key_field: The field to check for duplicates

        Returns:
        - Number of duplicates removed
        """
        try:
            collection = self.get_collection(collection_name)

            # Find duplicate key values
            pipeline = [
                {'$group': {'_id': f'${key_field}', 'count': {'$sum': 1}, 'docs': {'$push': '$_id'}}},
                {'$match': {'count': {'$gt': 1}}},
            ]

            duplicates = list(collection.aggregate(pipeline))

            removed_count = 0

            for dup in duplicates:
                key_value = dup['_id']
                if key_value is None:
                    continue  # Skip entries with null key values

                # Get all documents with this key value, sorted by created_at (newest first)
                docs = list(collection.find({key_field: key_value}).sort('created_at', -1))

                # Keep the newest document, remove others
                if len(docs) > 1:
                    # Keep the first (newest) document
                    keep_id = docs[0]['_id']

                    # Remove all other documents with this key
                    result = collection.delete_many({
                        key_field: key_value,
                        '_id': {'$ne': keep_id}  # Don't delete the one we're keeping
                    })

                    removed_count += result.deleted_count

            logging.info(f"Removed {removed_count} duplicate documents from {collection_name} collection.")
            return removed_count

        except Exception as e:
            logging.error(f"Error removing duplicates from {collection_name}: {e}")
            return 0