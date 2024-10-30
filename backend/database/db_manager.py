from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class Translation(Base):
    __tablename__ = 'translations'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    full_text = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class DatabaseManager:
    def __init__(self, db_url='sqlite:///asl_translations.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def store_translation(self, user_id, asl_text):
        session = self.Session()
        new_translation = Translation(user_id=user_id, full_text=asl_text)
        session.add(new_translation)
        session.commit()
        translation_id = new_translation.id
        session.close()
        return translation_id

    def get_translation(self, translation_id):
        session = self.Session()
        translation = session.query(Translation).filter_by(id=translation_id).first()
        session.close()
        return translation

db_manager = DatabaseManager()