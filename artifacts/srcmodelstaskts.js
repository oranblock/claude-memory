typescriptCopyimport mongoose, { Schema, Document } from 'mongoose';

/**
 * Interface for Task document
 */
export interface ITask extends Document {
  title: string;
  description?: string;
  status: 'not-started' | 'in-progress' | 'completed';
  priority: 'low' | 'medium' | 'high';
  dueDate?: Date;
  userId: mongoose.Types.ObjectId;
  categoryId: mongoose.Types.ObjectId;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Mongoose schema for Task
 */
const TaskSchema: Schema = new Schema(
  {
    title: {
      type: String,
      required: [true, 'Title is required'],
      trim: true,
      minlength: [2, 'Title must be at least 2 characters long'],
      maxlength: [100, 'Title cannot exceed 100 characters'],
    },
    description: {
      type: String,
      trim: true,
      maxlength: [500, 'Description cannot exceed 500 characters'],
    },
    status: {
      type: String,
      required: true,
      enum: {
        values: ['not-started', 'in-progress', 'completed'],
        message: 'Status must be not-started, in-progress, or completed',
      },
      default: 'not-started',
    },
    priority: {
      type: String,
      required: true,
      enum: {
        values: ['low', 'medium', 'high'],
        message: 'Priority must be low, medium, or high',
      },
      default: 'medium',
    },
    dueDate: {
      type: Date,
      validate: {
        validator: function (value: Date) {
          return !value || value >= new Date(Date.now() - 24 * 60 * 60 * 1000); // Allow dates from yesterday
        },
        message: 'Due date cannot be in the past',
      },
    },
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'User ID is required'],
    },
    categoryId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Category',
      required: [true, 'Category ID is required'],
    },
  },
  {
    timestamps: true,
  }
);

// Index for faster queries by user
TaskSchema.index({ userId: 1 });

// Compound index for user tasks by status
TaskSchema.index({ userId: 1, status: 1 });

// Compound index for user tasks by due date
TaskSchema.index({ userId: 1, dueDate: 1 });

export const Task = mongoose.model<ITask>('Task', TaskSchema);